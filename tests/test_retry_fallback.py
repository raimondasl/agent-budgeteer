"""Tests for Milestone 11B — Retry and fallback."""

import pytest

from budgeteer.config import BudgeteerConfig
from budgeteer.exceptions import RetryExhaustedError
from budgeteer.llm_client import LLMClient
from budgeteer.models import ModelTier, RunBudget, StepResult
from budgeteer.sdk import Budgeteer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _failing_llm_fn(fail_count=0):
    """Return an LLM callable that fails *fail_count* times then succeeds."""
    state = {"calls": 0}

    def fn(*, model, messages, max_tokens, temperature, **kwargs):
        state["calls"] += 1
        if state["calls"] <= fail_count:
            raise RuntimeError(f"LLM error on call {state['calls']}")
        return {
            "content": f"ok from {model}",
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            "model": model,
        }

    return fn, state


def _always_failing_llm_fn():
    def fn(*, model, messages, max_tokens, temperature, **kwargs):
        raise RuntimeError(f"LLM always fails for {model}")
    return fn


def _tier_config(tmp_db, max_retries=2, retry_delay_ms=0, fallback_enabled=True):
    return BudgeteerConfig(
        storage_path=tmp_db,
        default_max_tokens=1024,
        max_retries=max_retries,
        retry_delay_ms=retry_delay_ms,
        fallback_enabled=fallback_enabled,
        model_tiers=[
            ModelTier(name="expensive", cost_per_prompt_token=0.01,
                      cost_per_completion_token=0.03, max_context_window=8192),
            ModelTier(name="cheap", cost_per_prompt_token=0.001,
                      cost_per_completion_token=0.002, max_context_window=4096),
        ],
    )


# ---------------------------------------------------------------------------
# No retries (max_retries=0)
# ---------------------------------------------------------------------------


class TestNoRetries:
    def test_success_no_retry(self, tmp_db):
        fn, state = _failing_llm_fn(fail_count=0)
        config = BudgeteerConfig(storage_path=tmp_db, max_retries=0)
        sdk = Budgeteer(config, llm_client=LLMClient(fn))
        run = sdk.start_run()

        result = sdk.execute_step(run.run_id, messages=[{"role": "user", "content": "hi"}])
        assert result.llm_response.content.startswith("ok")
        assert state["calls"] == 1
        sdk.end_run(run.run_id)
        sdk.close()

    def test_failure_no_retry_raises_original(self, tmp_db):
        fn = _always_failing_llm_fn()
        config = BudgeteerConfig(storage_path=tmp_db, max_retries=0)
        sdk = Budgeteer(config, llm_client=LLMClient(fn))
        run = sdk.start_run()

        with pytest.raises(RuntimeError, match="LLM always fails"):
            sdk.execute_step(run.run_id, messages=[{"role": "user", "content": "hi"}])
        sdk.end_run(run.run_id)
        sdk.close()


# ---------------------------------------------------------------------------
# Retry logic
# ---------------------------------------------------------------------------


class TestRetry:
    def test_retry_succeeds_after_failure(self, tmp_db):
        fn, state = _failing_llm_fn(fail_count=1)
        config = BudgeteerConfig(
            storage_path=tmp_db, max_retries=2, retry_delay_ms=0
        )
        sdk = Budgeteer(config, llm_client=LLMClient(fn))
        run = sdk.start_run()

        result = sdk.execute_step(run.run_id, messages=[{"role": "user", "content": "hi"}])
        assert result.llm_response.content.startswith("ok")
        assert state["calls"] == 2  # 1 fail + 1 success
        sdk.end_run(run.run_id)
        sdk.close()

    def test_retry_succeeds_on_last_attempt(self, tmp_db):
        fn, state = _failing_llm_fn(fail_count=2)
        config = BudgeteerConfig(
            storage_path=tmp_db, max_retries=2, retry_delay_ms=0
        )
        sdk = Budgeteer(config, llm_client=LLMClient(fn))
        run = sdk.start_run()

        result = sdk.execute_step(run.run_id, messages=[{"role": "user", "content": "hi"}])
        assert state["calls"] == 3  # 2 fails + 1 success
        sdk.end_run(run.run_id)
        sdk.close()

    def test_all_retries_exhausted_raises(self, tmp_db):
        fn = _always_failing_llm_fn()
        config = _tier_config(tmp_db, max_retries=1, retry_delay_ms=0)
        sdk = Budgeteer(config, llm_client=LLMClient(fn))
        run = sdk.start_run(run_budget=RunBudget(hard_usd_cap=100.0))

        with pytest.raises(RetryExhaustedError) as exc_info:
            sdk.execute_step(run.run_id, messages=[{"role": "user", "content": "hi"}])

        err = exc_info.value
        assert err.attempts > 1
        assert isinstance(err.last_error, RuntimeError)
        assert len(err.models_tried) >= 1
        sdk.end_run(run.run_id)
        sdk.close()


# ---------------------------------------------------------------------------
# Fallback to cheaper model
# ---------------------------------------------------------------------------


class TestFallback:
    def test_fallback_to_cheaper_model(self, tmp_db):
        """When primary model fails, falls back to cheaper model."""
        state = {"calls": []}

        def fn(*, model, messages, max_tokens, temperature, **kwargs):
            state["calls"].append(model)
            if model == "expensive":
                raise RuntimeError("expensive model down")
            return {
                "content": f"ok from {model}",
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                "model": model,
            }

        config = _tier_config(tmp_db, max_retries=1, retry_delay_ms=0, fallback_enabled=True)
        sdk = Budgeteer(config, llm_client=LLMClient(fn))
        run = sdk.start_run(run_budget=RunBudget(hard_usd_cap=100.0))

        result = sdk.execute_step(run.run_id, messages=[{"role": "user", "content": "hi"}])

        # Should have tried expensive, then fallen back to cheap
        assert "cheap" in state["calls"]
        assert result.llm_response.content == "ok from cheap"
        sdk.end_run(run.run_id)
        sdk.close()

    def test_fallback_disabled(self, tmp_db):
        """With fallback_enabled=False, only retries primary model."""
        state = {"calls": []}

        def fn(*, model, messages, max_tokens, temperature, **kwargs):
            state["calls"].append(model)
            raise RuntimeError(f"fail {model}")

        config = _tier_config(tmp_db, max_retries=1, retry_delay_ms=0, fallback_enabled=False)
        sdk = Budgeteer(config, llm_client=LLMClient(fn))
        run = sdk.start_run(run_budget=RunBudget(hard_usd_cap=100.0))

        with pytest.raises(RetryExhaustedError):
            sdk.execute_step(run.run_id, messages=[{"role": "user", "content": "hi"}])

        # Should only have tried the selected model (no fallback)
        unique_models = set(state["calls"])
        assert len(unique_models) == 1
        sdk.end_run(run.run_id)
        sdk.close()

    def test_no_fallback_without_tiers(self, tmp_db):
        """Without model_tiers, no fallback models available."""
        state = {"calls": []}

        def fn(*, model, messages, max_tokens, temperature, **kwargs):
            state["calls"].append(model)
            raise RuntimeError("fail")

        config = BudgeteerConfig(
            storage_path=tmp_db, max_retries=1, retry_delay_ms=0
        )
        sdk = Budgeteer(config, llm_client=LLMClient(fn))
        run = sdk.start_run()

        with pytest.raises(RetryExhaustedError):
            sdk.execute_step(run.run_id, messages=[{"role": "user", "content": "hi"}])

        # Only tried the default model
        assert all(m == "gpt-4o-mini" for m in state["calls"])
        sdk.end_run(run.run_id)
        sdk.close()


# ---------------------------------------------------------------------------
# RetryExhaustedError
# ---------------------------------------------------------------------------


class TestRetryExhaustedError:
    def test_error_attributes(self):
        inner = ValueError("inner error")
        err = RetryExhaustedError(
            attempts=5, last_error=inner, models_tried=["m1", "m2"]
        )
        assert err.attempts == 5
        assert err.last_error is inner
        assert err.models_tried == ["m1", "m2"]
        assert "5" in str(err)
        assert "m1" in str(err)

    def test_error_without_models(self):
        err = RetryExhaustedError(attempts=1, last_error=RuntimeError("x"))
        assert err.models_tried == []
        assert "unknown" in str(err)


# ---------------------------------------------------------------------------
# Config fields
# ---------------------------------------------------------------------------


class TestRetryConfig:
    def test_defaults(self):
        config = BudgeteerConfig()
        assert config.max_retries == 0
        assert config.retry_delay_ms == 1000.0
        assert config.fallback_enabled is True

    def test_from_dict(self):
        data = {"max_retries": 3, "retry_delay_ms": 500, "fallback_enabled": False}
        config = BudgeteerConfig.from_dict(data)
        assert config.max_retries == 3
        assert config.retry_delay_ms == 500
        assert config.fallback_enabled is False
