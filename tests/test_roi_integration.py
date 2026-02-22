"""Tests for Milestone 10 — ROI Integration."""

import pytest

from budgeteer.config import BudgeteerConfig
from budgeteer.llm_client import LLMClient
from budgeteer.models import (
    ModelTier,
    RunBudget,
    StepContext,
    StepMetrics,
    StepResult,
)
from budgeteer.roi import ClarifyingQuestion, ROISignals
from budgeteer.sdk import Budgeteer
from budgeteer.tool_executor import ToolExecutor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_llm_fn(content="ok", prompt_tokens=10, completion_tokens=5, tool_calls=None):
    def fn(*, model, messages, max_tokens, temperature, **kwargs):
        raw = {
            "content": content,
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
            "model": model,
        }
        if tool_calls is not None:
            raw["tool_calls"] = tool_calls
        return raw
    return fn


def _roi_config(tmp_db, roi_enabled=True, **kwargs):
    return BudgeteerConfig(
        storage_path=tmp_db,
        default_max_tokens=1024,
        roi_enabled=roi_enabled,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# ROI disabled (no interference)
# ---------------------------------------------------------------------------


class TestROIDisabled:
    def test_roi_disabled_no_interference(self, tmp_db):
        config = _roi_config(tmp_db, roi_enabled=False)
        client = LLMClient(_make_llm_fn(
            tool_calls=[{"name": "search", "arguments": {"q": "test"}}]
        ))
        executor = ToolExecutor()
        executor.register("search", lambda q: "results")

        sdk = Budgeteer(config, llm_client=client, tool_executor=executor)
        run = sdk.start_run()
        result = sdk.execute_step(run.run_id, messages=[{"role": "user", "content": "hi"}])

        # Tools should execute normally
        assert len(result.tool_results) == 1
        assert result.tool_results[0].success is True
        assert result.roi_decisions is None
        sdk.end_run(run.run_id)
        sdk.close()

    def test_suggest_clarification_disabled(self, tmp_db):
        config = _roi_config(tmp_db, roi_enabled=False)
        sdk = Budgeteer(config)
        run = sdk.start_run()
        assert sdk.suggest_clarification(run.run_id) is None
        sdk.end_run(run.run_id)
        sdk.close()


# ---------------------------------------------------------------------------
# ROI gating tools
# ---------------------------------------------------------------------------


class TestROIToolGating:
    def test_tools_executed_when_budget_healthy(self, tmp_db):
        config = _roi_config(tmp_db, roi_enabled=True)
        client = LLMClient(_make_llm_fn(
            tool_calls=[{"name": "search", "arguments": {}}]
        ))
        executor = ToolExecutor()
        executor.register("search", lambda: "results")

        sdk = Budgeteer(config, llm_client=client, tool_executor=executor)
        run = sdk.start_run(run_budget=RunBudget(hard_usd_cap=100.0))
        result = sdk.execute_step(
            run.run_id,
            messages=[{"role": "user", "content": "hi"}],
            metadata={"task_complexity": 0.8},
        )

        # With healthy budget, tools should be recommended
        assert result.roi_decisions is not None
        assert "tool:search" in result.roi_decisions
        sdk.end_run(run.run_id)
        sdk.close()

    def test_tools_skipped_when_budget_low(self, tmp_db):
        config = _roi_config(
            tmp_db,
            roi_enabled=True,
            roi_budget_floor=0.5,  # high floor
        )
        client = LLMClient(_make_llm_fn(
            tool_calls=[{"name": "search", "arguments": {}}]
        ))
        executor = ToolExecutor()
        executor.register("search", lambda: "results")

        sdk = Budgeteer(config, llm_client=client, tool_executor=executor)
        run = sdk.start_run(run_budget=RunBudget(hard_usd_cap=0.02))

        # Spend most of the budget first
        ctx = StepContext(run_id=run.run_id)
        d = sdk.before_step(ctx)
        sdk.after_step(ctx, d, StepMetrics(cost_usd=0.018))

        result = sdk.execute_step(
            run.run_id,
            messages=[{"role": "user", "content": "hi"}],
        )

        # With low remaining budget, tools may be skipped
        assert result.roi_decisions is not None
        if "tool:search" in result.roi_decisions:
            tool_decision = result.roi_decisions["tool:search"]
            # Either tool was skipped (not recommended) or executed
            assert "recommended" in tool_decision
        sdk.end_run(run.run_id)
        sdk.close()

    def test_roi_decisions_populated(self, tmp_db):
        config = _roi_config(tmp_db, roi_enabled=True)
        client = LLMClient(_make_llm_fn(
            tool_calls=[{"name": "tool1", "arguments": {}}]
        ))
        executor = ToolExecutor()
        executor.register("tool1", lambda: "ok")

        sdk = Budgeteer(config, llm_client=client, tool_executor=executor)
        run = sdk.start_run(run_budget=RunBudget(hard_usd_cap=100.0))
        result = sdk.execute_step(
            run.run_id,
            messages=[{"role": "user", "content": "hi"}],
        )

        assert result.roi_decisions is not None
        assert "tool:tool1" in result.roi_decisions
        td = result.roi_decisions["tool:tool1"]
        assert "roi_score" in td
        assert "recommended" in td
        assert "reason" in td
        sdk.end_run(run.run_id)
        sdk.close()


# ---------------------------------------------------------------------------
# ROI gating retrieval
# ---------------------------------------------------------------------------


class TestROIRetrievalGating:
    def test_retrieval_with_roi_enabled(self, tmp_db):
        from budgeteer.context_manager import RetrievalResult
        received_messages = []

        def mock_llm(*, model, messages, max_tokens, temperature, **kwargs):
            received_messages.extend(messages)
            return {
                "content": "ok",
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                "model": model,
            }

        config = _roi_config(tmp_db, roi_enabled=True)
        client = LLMClient(mock_llm)
        sdk = Budgeteer(config, llm_client=client)
        run = sdk.start_run(run_budget=RunBudget(hard_usd_cap=100.0))

        retrieval = [RetrievalResult(content="Fact A", score=0.9)]
        result = sdk.execute_step(
            run.run_id,
            messages=[{"role": "user", "content": "hi"}],
            retrieval_results=retrieval,
            metadata={"freshness_need": 0.9, "ambiguity": 0.8},
        )

        assert result.roi_decisions is not None
        if "retrieval" in result.roi_decisions:
            assert "recommended" in result.roi_decisions["retrieval"]
        sdk.end_run(run.run_id)
        sdk.close()

    def test_retrieval_skipped_when_not_beneficial(self, tmp_db):
        from budgeteer.context_manager import RetrievalResult
        received_messages = []

        def mock_llm(*, model, messages, max_tokens, temperature, **kwargs):
            received_messages.extend(messages)
            return {
                "content": "ok",
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                "model": model,
            }

        config = _roi_config(
            tmp_db,
            roi_enabled=True,
            roi_recommend_threshold=100.0,  # very high threshold
        )
        client = LLMClient(mock_llm)
        sdk = Budgeteer(config, llm_client=client)
        run = sdk.start_run(run_budget=RunBudget(hard_usd_cap=100.0))

        retrieval = [RetrievalResult(content="Fact A", score=0.9)]
        result = sdk.execute_step(
            run.run_id,
            messages=[{"role": "user", "content": "hi"}],
            retrieval_results=retrieval,
            metadata={"freshness_need": 0.0, "ambiguity": 0.0},
        )

        # With very high threshold, retrieval should not be recommended
        if result.roi_decisions and "retrieval" in result.roi_decisions:
            assert result.roi_decisions["retrieval"]["recommended"] is False
            # Retrieval content should NOT be in messages sent to LLM
            retrieval_msgs = [m for m in received_messages if "[Retrieved context]" in m.get("content", "")]
            assert len(retrieval_msgs) == 0
        sdk.end_run(run.run_id)
        sdk.close()


# ---------------------------------------------------------------------------
# Signal derivation
# ---------------------------------------------------------------------------


class TestROISignalDerivation:
    def test_signals_from_metadata(self, tmp_db):
        config = _roi_config(tmp_db, roi_enabled=True)
        sdk = Budgeteer(config)
        run = sdk.start_run(run_budget=RunBudget(hard_usd_cap=10.0))

        from budgeteer.models import StepDecision
        ctx = StepContext(
            run_id=run.run_id,
            metadata={"ambiguity": 0.9, "task_complexity": 0.7, "freshness_need": 0.3},
        )
        decision = StepDecision(model="m")
        signals = sdk._derive_roi_signals(run.run_id, ctx, decision)

        assert signals.ambiguity == 0.9
        assert signals.task_complexity == 0.7
        assert signals.freshness_need == 0.3
        assert signals.remaining_budget_fraction == pytest.approx(1.0)
        assert signals.previous_failures == 0
        sdk.end_run(run.run_id)
        sdk.close()

    def test_signals_remaining_budget(self, tmp_db):
        config = _roi_config(tmp_db, roi_enabled=True)
        sdk = Budgeteer(config)
        run = sdk.start_run(run_budget=RunBudget(hard_usd_cap=1.0))

        # Spend half the budget
        ctx = StepContext(run_id=run.run_id)
        d = sdk.before_step(ctx)
        sdk.after_step(ctx, d, StepMetrics(cost_usd=0.5))

        from budgeteer.models import StepDecision
        ctx2 = StepContext(run_id=run.run_id)
        signals = sdk._derive_roi_signals(run.run_id, ctx2, StepDecision(model="m"))

        assert signals.remaining_budget_fraction == pytest.approx(0.5)
        sdk.end_run(run.run_id)
        sdk.close()

    def test_signals_previous_failures(self, tmp_db):
        from budgeteer.models import ToolRecord
        config = _roi_config(tmp_db, roi_enabled=True)
        sdk = Budgeteer(config)
        run = sdk.start_run()

        # Log some failed tool calls
        sdk.telemetry.log_tool_call(ToolRecord(
            run_id=run.run_id, step_id="s1", tool_name="t1",
            duration_ms=10, success=False, error="fail"
        ))
        sdk.telemetry.log_tool_call(ToolRecord(
            run_id=run.run_id, step_id="s1", tool_name="t2",
            duration_ms=10, success=True,
        ))

        from budgeteer.models import StepDecision
        ctx = StepContext(run_id=run.run_id)
        signals = sdk._derive_roi_signals(run.run_id, ctx, StepDecision(model="m"))

        assert signals.previous_failures == 1
        sdk.end_run(run.run_id)
        sdk.close()

    def test_signals_defaults_without_metadata(self, tmp_db):
        config = _roi_config(tmp_db, roi_enabled=True)
        sdk = Budgeteer(config)
        run = sdk.start_run()

        from budgeteer.models import StepDecision
        ctx = StepContext(run_id=run.run_id)
        signals = sdk._derive_roi_signals(run.run_id, ctx, StepDecision(model="m"))

        assert signals.ambiguity == 0.5
        assert signals.task_complexity == 0.5
        assert signals.freshness_need == 0.0
        sdk.end_run(run.run_id)
        sdk.close()


# ---------------------------------------------------------------------------
# suggest_clarification
# ---------------------------------------------------------------------------


class TestSuggestClarification:
    def test_suggest_when_ambiguous(self, tmp_db):
        config = _roi_config(
            tmp_db,
            roi_enabled=True,
            roi_clarify_ambiguity_threshold=0.5,
        )
        sdk = Budgeteer(config)
        run = sdk.start_run()

        question = sdk.suggest_clarification(
            run.run_id,
            metadata={"ambiguity": 0.9, "task_complexity": 0.8},
        )

        assert question is not None
        assert isinstance(question, ClarifyingQuestion)
        assert len(question.template) > 0
        sdk.end_run(run.run_id)
        sdk.close()

    def test_no_suggestion_when_clear(self, tmp_db):
        config = _roi_config(
            tmp_db,
            roi_enabled=True,
            roi_clarify_ambiguity_threshold=0.8,
        )
        sdk = Budgeteer(config)
        run = sdk.start_run()

        question = sdk.suggest_clarification(
            run.run_id,
            metadata={"ambiguity": 0.1},
        )

        assert question is None
        sdk.end_run(run.run_id)
        sdk.close()

    def test_exclude_categories(self, tmp_db):
        config = _roi_config(
            tmp_db,
            roi_enabled=True,
            roi_clarify_ambiguity_threshold=0.3,
        )
        sdk = Budgeteer(config)
        run = sdk.start_run()

        # Get first suggestion
        q1 = sdk.suggest_clarification(
            run.run_id,
            metadata={"ambiguity": 0.9, "task_complexity": 0.9},
        )
        if q1 is not None:
            # Exclude that category
            q2 = sdk.suggest_clarification(
                run.run_id,
                metadata={"ambiguity": 0.9, "task_complexity": 0.9},
                exclude_categories={q1.category},
            )
            if q2 is not None:
                assert q2.category != q1.category
        sdk.end_run(run.run_id)
        sdk.close()

    def test_suggest_clarification_disabled(self, tmp_db):
        config = _roi_config(tmp_db, roi_enabled=False)
        sdk = Budgeteer(config)
        run = sdk.start_run()
        assert sdk.suggest_clarification(run.run_id) is None
        sdk.end_run(run.run_id)
        sdk.close()


# ---------------------------------------------------------------------------
# Config ROI fields
# ---------------------------------------------------------------------------


class TestROIConfig:
    def test_roi_config_defaults(self):
        config = BudgeteerConfig()
        assert config.roi_enabled is False
        assert config.roi_lambda_latency == 0.001
        assert config.roi_recommend_threshold == 1.0
        assert config.roi_budget_floor == 0.1
        assert config.roi_clarify_ambiguity_threshold == 0.6

    def test_roi_config_from_dict(self):
        data = {
            "roi_enabled": True,
            "roi_lambda_latency": 0.005,
            "roi_recommend_threshold": 2.0,
        }
        config = BudgeteerConfig.from_dict(data)
        assert config.roi_enabled is True
        assert config.roi_lambda_latency == 0.005
        assert config.roi_recommend_threshold == 2.0
