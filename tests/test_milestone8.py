"""Tests for Milestone 8 — Reporter + Cost Utilities."""

import pytest

from budgeteer.config import BudgeteerConfig
from budgeteer.llm_client import LLMClient
from budgeteer.models import (
    LLMResponse,
    ModelTier,
    RunBudget,
    StepContext,
    StepMetrics,
    compute_cost,
)
from budgeteer.reporting import FullReport
from budgeteer.sdk import Budgeteer
from budgeteer.telemetry import TelemetryStore


# ---------------------------------------------------------------------------
# compute_cost utility
# ---------------------------------------------------------------------------


class TestComputeCost:
    def test_basic_cost(self):
        tiers = [
            ModelTier(name="m1", cost_per_prompt_token=0.01, cost_per_completion_token=0.03, max_context_window=4096),
        ]
        cost = compute_cost("m1", prompt_tokens=100, completion_tokens=50, model_tiers=tiers)
        assert cost == pytest.approx(100 * 0.01 + 50 * 0.03)

    def test_model_not_found(self):
        tiers = [
            ModelTier(name="m1", cost_per_prompt_token=0.01, cost_per_completion_token=0.03, max_context_window=4096),
        ]
        assert compute_cost("unknown", 100, 50, tiers) == 0.0

    def test_empty_tiers(self):
        assert compute_cost("m1", 100, 50, []) == 0.0

    def test_zero_tokens(self):
        tiers = [
            ModelTier(name="m1", cost_per_prompt_token=0.01, cost_per_completion_token=0.03, max_context_window=4096),
        ]
        assert compute_cost("m1", 0, 0, tiers) == 0.0


# ---------------------------------------------------------------------------
# LLMClient cost tracking
# ---------------------------------------------------------------------------

def _make_callable(content="Hello", prompt_tokens=10, completion_tokens=5, model=None):
    def fn(*, model: str, messages, max_tokens, temperature, **kwargs):
        resp = {
            "content": content,
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
        }
        if model is not None:
            resp["model"] = model
        return resp
    return fn


class TestLLMClientCost:
    def test_cost_computed_with_tiers(self):
        tiers = [
            ModelTier(name="gpt-4o", cost_per_prompt_token=0.01, cost_per_completion_token=0.03, max_context_window=8192),
        ]
        client = LLMClient(_make_callable(prompt_tokens=100, completion_tokens=50, model="gpt-4o"), model_tiers=tiers)
        resp = client.complete("gpt-4o", [])
        expected_cost = 100 * 0.01 + 50 * 0.03
        assert resp.cost_usd == pytest.approx(expected_cost)

    def test_cost_zero_without_tiers(self):
        client = LLMClient(_make_callable(prompt_tokens=100, completion_tokens=50))
        resp = client.complete("m", [])
        assert resp.cost_usd == 0.0

    def test_total_cost_accumulates(self):
        tiers = [
            ModelTier(name="m", cost_per_prompt_token=0.001, cost_per_completion_token=0.002, max_context_window=4096),
        ]

        def fn(*, model, messages, max_tokens, temperature, **kwargs):
            return {"content": "", "usage": {"prompt_tokens": 10, "completion_tokens": 5}, "model": "m"}

        client = LLMClient(fn, model_tiers=tiers)
        client.complete("m", [])
        client.complete("m", [])
        expected = 2 * (10 * 0.001 + 5 * 0.002)
        assert client.total_cost_usd == pytest.approx(expected)

    def test_cost_usd_on_llm_response(self):
        resp = LLMResponse(content="hi", model="m", cost_usd=0.5)
        assert resp.cost_usd == 0.5

    def test_llm_response_default_cost(self):
        resp = LLMResponse(content="hi", model="m")
        assert resp.cost_usd == 0.0


# ---------------------------------------------------------------------------
# TelemetryStore.list_run_ids
# ---------------------------------------------------------------------------


class TestListRunIds:
    def test_list_empty(self, tmp_db):
        store = TelemetryStore(tmp_db)
        assert store.list_run_ids() == []
        store.close()

    def test_list_returns_ids(self, tmp_db):
        from budgeteer.models import RunRecord
        store = TelemetryStore(tmp_db)
        store.log_run(RunRecord(run_id="r1", start_time=1.0))
        store.log_run(RunRecord(run_id="r2", start_time=2.0))
        ids = store.list_run_ids()
        assert ids == ["r1", "r2"]
        store.close()

    def test_list_ordered_by_start_time(self, tmp_db):
        from budgeteer.models import RunRecord
        store = TelemetryStore(tmp_db)
        store.log_run(RunRecord(run_id="late", start_time=100.0))
        store.log_run(RunRecord(run_id="early", start_time=1.0))
        ids = store.list_run_ids()
        assert ids == ["early", "late"]
        store.close()


# ---------------------------------------------------------------------------
# SDK.report()
# ---------------------------------------------------------------------------


class TestSDKReporting:
    def test_report_empty(self, tmp_db):
        config = BudgeteerConfig(storage_path=tmp_db)
        sdk = Budgeteer(config)
        report = sdk.report()
        assert isinstance(report, FullReport)
        assert report.run_summaries == []
        sdk.close()

    def test_report_with_runs(self, tmp_db):
        config = BudgeteerConfig(storage_path=tmp_db)
        sdk = Budgeteer(config)
        run = sdk.start_run()
        ctx = StepContext(run_id=run.run_id)
        d = sdk.before_step(ctx)
        sdk.after_step(ctx, d, StepMetrics(cost_usd=0.05, prompt_tokens=100, completion_tokens=50))
        sdk.end_run(run.run_id, success=True)

        report = sdk.report()
        assert len(report.run_summaries) == 1
        assert report.run_summaries[0].run_id == run.run_id
        assert report.run_summaries[0].total_cost_usd == pytest.approx(0.05)
        sdk.close()

    def test_report_specific_run_ids(self, tmp_db):
        config = BudgeteerConfig(storage_path=tmp_db)
        sdk = Budgeteer(config)
        r1 = sdk.start_run()
        sdk.end_run(r1.run_id)
        r2 = sdk.start_run()
        sdk.end_run(r2.run_id)

        report = sdk.report(run_ids=[r1.run_id])
        assert len(report.run_summaries) == 1
        assert report.run_summaries[0].run_id == r1.run_id
        sdk.close()

    def test_report_with_budget_caps(self, tmp_db):
        config = BudgeteerConfig(storage_path=tmp_db)
        sdk = Budgeteer(config)
        run = sdk.start_run()
        ctx = StepContext(run_id=run.run_id)
        d = sdk.before_step(ctx)
        sdk.after_step(ctx, d, StepMetrics(cost_usd=2.0, prompt_tokens=100, completion_tokens=50))
        sdk.end_run(run.run_id, success=True)

        report = sdk.report(budget_caps={"usd": 1.0})
        assert report.budget_compliance.runs_exceeded == 1
        assert report.budget_compliance.compliance_rate == 0.0
        sdk.close()

    def test_report_auto_discovers_all_runs(self, tmp_db):
        config = BudgeteerConfig(storage_path=tmp_db)
        sdk = Budgeteer(config)
        for _ in range(3):
            run = sdk.start_run()
            sdk.end_run(run.run_id)

        report = sdk.report()
        assert len(report.run_summaries) == 3
        sdk.close()
