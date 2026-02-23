"""Tests for logging and event hooks (Milestone 14)."""

from __future__ import annotations

import logging

import pytest

from budgeteer.config import BudgeteerConfig
from budgeteer.events import (
    BUDGET_WARNING,
    FALLBACK_TRIGGERED,
    RETRY_ATTEMPT,
    RUN_ENDED,
    RUN_STARTED,
    STEP_COMPLETED,
    STEP_DECIDED,
    BudgeteerEvent,
)
from budgeteer.models import (
    ModelTier,
    RunBudget,
    StepContext,
    StepDecision,
    StepMetrics,
)
from budgeteer.sdk import Budgeteer


def _make_budgeteer(tmp_path, **kwargs):
    """Helper to create a Budgeteer with tmp storage."""
    cfg = BudgeteerConfig(storage_path=str(tmp_path / "tel.db"), **kwargs)
    return Budgeteer(config=cfg)


class TestEventHookRegistration:
    """Tests for add_hook / remove_hook."""

    def test_add_hook(self, tmp_path):
        b = _make_budgeteer(tmp_path)
        events = []
        b.add_hook(events.append)
        b.start_run()
        assert len(events) >= 1
        b.close()

    def test_remove_hook(self, tmp_path):
        b = _make_budgeteer(tmp_path)
        events = []
        b.add_hook(events.append)
        b.remove_hook(events.append)
        b.start_run()
        assert len(events) == 0
        b.close()

    def test_multiple_hooks_all_called(self, tmp_path):
        b = _make_budgeteer(tmp_path)
        events1 = []
        events2 = []
        b.add_hook(events1.append)
        b.add_hook(events2.append)
        b.start_run()
        assert len(events1) >= 1
        assert len(events2) >= 1
        b.close()


class TestHookExceptionSafety:
    """Hook exceptions should not crash the SDK."""

    def test_bad_hook_does_not_crash(self, tmp_path):
        b = _make_budgeteer(tmp_path)

        def bad_hook(event):
            raise RuntimeError("hook failure")

        good_events = []
        b.add_hook(bad_hook)
        b.add_hook(good_events.append)
        b.start_run()  # should not raise
        assert len(good_events) >= 1
        b.close()


class TestRunLifecycleEvents:
    """Events emitted during run start/end."""

    def test_run_started_event(self, tmp_path):
        b = _make_budgeteer(tmp_path)
        events = []
        b.add_hook(events.append)
        run = b.start_run(run_budget=RunBudget(hard_usd_cap=1.0))
        started = [e for e in events if e.event_type == RUN_STARTED]
        assert len(started) == 1
        assert started[0].run_id == run.run_id
        assert started[0].data["hard_usd_cap"] == 1.0
        b.close()

    def test_run_ended_event(self, tmp_path):
        b = _make_budgeteer(tmp_path)
        events = []
        b.add_hook(events.append)
        run = b.start_run()
        b.end_run(run.run_id, success=True)
        ended = [e for e in events if e.event_type == RUN_ENDED]
        assert len(ended) == 1
        assert ended[0].data["success"] is True
        b.close()


class TestStepEvents:
    """Events emitted during before_step / after_step."""

    def test_step_decided_event(self, tmp_path):
        b = _make_budgeteer(tmp_path)
        events = []
        b.add_hook(events.append)
        run = b.start_run()
        ctx = StepContext(run_id=run.run_id)
        b.before_step(ctx)
        decided = [e for e in events if e.event_type == STEP_DECIDED]
        assert len(decided) == 1
        assert decided[0].data["model"] == "gpt-4o-mini"
        b.close()

    def test_step_completed_event(self, tmp_path):
        b = _make_budgeteer(tmp_path)
        events = []
        b.add_hook(events.append)
        run = b.start_run()
        ctx = StepContext(run_id=run.run_id)
        decision = b.before_step(ctx)
        metrics = StepMetrics(prompt_tokens=100, completion_tokens=50, cost_usd=0.005, latency_ms=200)
        b.after_step(ctx, decision, metrics)
        completed = [e for e in events if e.event_type == STEP_COMPLETED]
        assert len(completed) == 1
        assert completed[0].data["cost_usd"] == 0.005
        assert completed[0].data["tokens"] == 150
        b.close()


class TestBudgetWarningEvents:
    """Events for degradation warnings."""

    def test_budget_warning_on_degradation(self, tmp_path):
        """When budget is tight (legacy path), a BUDGET_WARNING event should be emitted."""
        # No model_tiers -> uses legacy degradation path
        b = _make_budgeteer(tmp_path)
        events = []
        b.add_hook(events.append)

        run = b.start_run(run_budget=RunBudget(hard_usd_cap=0.01))
        # Simulate spending 85% of budget -> triggers level 1 degradation
        b._active_runs[run.run_id].total_cost_usd = 0.0085

        ctx = StepContext(run_id=run.run_id)
        b.before_step(ctx)
        warnings = [e for e in events if e.event_type == BUDGET_WARNING]
        assert len(warnings) >= 1
        b.close()


class TestEventDataFields:
    """Verify event data contains expected fields."""

    def test_run_started_has_budget_caps(self, tmp_path):
        b = _make_budgeteer(tmp_path)
        events = []
        b.add_hook(events.append)
        b.start_run(run_budget=RunBudget(hard_usd_cap=2.0, hard_token_cap=5000))
        started = [e for e in events if e.event_type == RUN_STARTED][0]
        assert "hard_usd_cap" in started.data
        assert "hard_token_cap" in started.data
        b.close()

    def test_step_decided_has_model_and_degrade(self, tmp_path):
        b = _make_budgeteer(tmp_path)
        events = []
        b.add_hook(events.append)
        run = b.start_run()
        ctx = StepContext(run_id=run.run_id)
        b.before_step(ctx)
        decided = [e for e in events if e.event_type == STEP_DECIDED][0]
        assert "model" in decided.data
        assert "degrade_level" in decided.data
        assert "max_tokens" in decided.data
        b.close()

    def test_event_has_timestamp(self, tmp_path):
        b = _make_budgeteer(tmp_path)
        events = []
        b.add_hook(events.append)
        b.start_run()
        assert events[0].timestamp > 0
        b.close()


class TestLogging:
    """Tests for Python logging output."""

    def test_start_run_logs_info(self, tmp_path, caplog):
        b = _make_budgeteer(tmp_path)
        with caplog.at_level(logging.INFO, logger="budgeteer"):
            b.start_run()
        assert "Run started" in caplog.text
        b.close()

    def test_end_run_logs_info(self, tmp_path, caplog):
        b = _make_budgeteer(tmp_path)
        run = b.start_run()
        with caplog.at_level(logging.INFO, logger="budgeteer"):
            b.end_run(run.run_id, success=True)
        assert "Run ended" in caplog.text
        b.close()

    def test_before_step_logs_debug(self, tmp_path, caplog):
        b = _make_budgeteer(tmp_path)
        run = b.start_run()
        ctx = StepContext(run_id=run.run_id)
        with caplog.at_level(logging.DEBUG, logger="budgeteer"):
            b.before_step(ctx)
        assert "Step decided" in caplog.text
        b.close()

    def test_after_step_logs_info(self, tmp_path, caplog):
        b = _make_budgeteer(tmp_path)
        run = b.start_run()
        ctx = StepContext(run_id=run.run_id)
        decision = b.before_step(ctx)
        metrics = StepMetrics(cost_usd=0.01, latency_ms=300)
        with caplog.at_level(logging.INFO, logger="budgeteer"):
            b.after_step(ctx, decision, metrics)
        assert "Step completed" in caplog.text
        b.close()

    def test_degradation_logs_warning(self, tmp_path, caplog):
        """Budget degradation should log at WARNING level (legacy path)."""
        # No model_tiers -> legacy degradation path
        b = _make_budgeteer(tmp_path)
        run = b.start_run(run_budget=RunBudget(hard_usd_cap=0.01))
        b._active_runs[run.run_id].total_cost_usd = 0.009  # 90% consumed
        ctx = StepContext(run_id=run.run_id)
        with caplog.at_level(logging.WARNING, logger="budgeteer"):
            b.before_step(ctx)
        assert "Degradation active" in caplog.text
        b.close()

    def test_bad_hook_logs_exception(self, tmp_path, caplog):
        b = _make_budgeteer(tmp_path)

        def bad_hook(event):
            raise ValueError("oops")

        b.add_hook(bad_hook)
        with caplog.at_level(logging.ERROR, logger="budgeteer"):
            b.start_run()
        assert "hook raised an exception" in caplog.text
        b.close()


class TestRetryFallbackEvents:
    """Events for retry and fallback scenarios."""

    def test_retry_event_emitted(self, tmp_path):
        from budgeteer.llm_client import LLMClient

        call_count = 0

        def failing_then_ok(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 1:
                raise RuntimeError("temporary failure")
            return {"content": "ok", "usage": {"prompt_tokens": 10, "completion_tokens": 5}, "model": "m"}

        tiers = [ModelTier(name="m", cost_per_prompt_token=0.001, cost_per_completion_token=0.002, max_context_window=4096)]
        client = LLMClient(call_fn=failing_then_ok, model_tiers=tiers)
        b = Budgeteer(
            config=BudgeteerConfig(
                storage_path=str(tmp_path / "tel.db"),
                model_tiers=tiers,
                max_retries=2,
                retry_delay_ms=0,
            ),
            llm_client=client,
        )
        events = []
        b.add_hook(events.append)
        run = b.start_run()
        b.execute_step(run.run_id, messages=[{"role": "user", "content": "hi"}])
        retries = [e for e in events if e.event_type == RETRY_ATTEMPT]
        assert len(retries) == 1
        assert retries[0].data["attempt"] == 1
        b.close()

    def test_fallback_event_emitted(self, tmp_path):
        from budgeteer.llm_client import LLMClient

        def always_fail_primary(**kwargs):
            if kwargs.get("model") == "expensive":
                raise RuntimeError("fail")
            return {"content": "ok", "usage": {"prompt_tokens": 10, "completion_tokens": 5}, "model": kwargs["model"]}

        tiers = [
            ModelTier(name="cheap", cost_per_prompt_token=0.0001, cost_per_completion_token=0.0002, max_context_window=4096),
            ModelTier(name="expensive", cost_per_prompt_token=0.01, cost_per_completion_token=0.03, max_context_window=8192),
        ]
        client = LLMClient(call_fn=always_fail_primary, model_tiers=tiers)
        b = Budgeteer(
            config=BudgeteerConfig(
                storage_path=str(tmp_path / "tel.db"),
                model_tiers=tiers,
                max_retries=1,
                retry_delay_ms=0,
                fallback_enabled=True,
            ),
            llm_client=client,
        )
        events = []
        b.add_hook(events.append)
        run = b.start_run()
        b.execute_step(run.run_id, messages=[{"role": "user", "content": "hi"}])
        fallbacks = [e for e in events if e.event_type == FALLBACK_TRIGGERED]
        assert len(fallbacks) >= 1
        b.close()
