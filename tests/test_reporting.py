"""Tests for budgeteer.reporting — run summaries, model stats, prediction accuracy, compliance."""

import pytest

from budgeteer.models import RunRecord, StepDecision, StepMetrics, StepRecord
from budgeteer.reporting import (
    BudgetComplianceReport,
    FullReport,
    ModelStats,
    PredictionAccuracy,
    Reporter,
    RunSummary,
)
from budgeteer.telemetry import TelemetryStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_store() -> TelemetryStore:
    """In-memory telemetry store."""
    return TelemetryStore(db_path=":memory:")


def _run(run_id="r1", cost=0.05, tokens=500, latency=2000.0, steps=3,
         tool_calls=5, success=True) -> RunRecord:
    return RunRecord(
        run_id=run_id,
        total_cost_usd=cost,
        total_tokens=tokens,
        total_latency_ms=latency,
        total_steps=steps,
        total_tool_calls=tool_calls,
        success=success,
    )


def _step(run_id="r1", step_id="s1", model="gpt-4o",
          pred=None, actual=None) -> StepRecord:
    return StepRecord(
        run_id=run_id,
        step_id=step_id,
        decision=StepDecision(model=model),
        predicted=pred,
        actual=actual,
    )


def _metrics(pt=100, ct=200, cost=0.01, latency=500.0) -> StepMetrics:
    return StepMetrics(
        prompt_tokens=pt,
        completion_tokens=ct,
        cost_usd=cost,
        latency_ms=latency,
    )


# ===========================================================================
# RunSummary dataclass
# ===========================================================================


class TestRunSummary:
    def test_defaults(self):
        s = RunSummary(run_id="r1")
        assert s.total_cost_usd == 0.0
        assert s.success is None

    def test_fields(self):
        s = RunSummary(run_id="r1", total_cost_usd=1.5, total_steps=10)
        assert s.total_cost_usd == 1.5
        assert s.total_steps == 10


# ===========================================================================
# Reporter.run_summary
# ===========================================================================


class TestReporterRunSummary:
    def test_returns_none_for_missing_run(self):
        store = _make_store()
        reporter = Reporter(store)
        assert reporter.run_summary("nonexistent") is None

    def test_returns_summary_for_logged_run(self):
        store = _make_store()
        store.log_run(_run(run_id="r1", cost=0.05, tokens=500))
        reporter = Reporter(store)
        s = reporter.run_summary("r1")
        assert s is not None
        assert s.run_id == "r1"
        assert s.total_cost_usd == 0.05
        assert s.total_tokens == 500
        assert s.success is True


# ===========================================================================
# Reporter.model_stats
# ===========================================================================


class TestModelStats:
    def test_empty_records(self):
        reporter = Reporter(_make_store())
        assert reporter.model_stats([]) == []

    def test_single_model_single_step(self):
        reporter = Reporter(_make_store())
        steps = [
            _step(model="gpt-4o", actual=_metrics(pt=100, ct=200, cost=0.01, latency=500)),
        ]
        stats = reporter.model_stats(steps)
        assert len(stats) == 1
        assert stats[0].model == "gpt-4o"
        assert stats[0].step_count == 1
        assert stats[0].total_cost_usd == 0.01
        assert stats[0].total_prompt_tokens == 100
        assert stats[0].total_completion_tokens == 200
        assert stats[0].avg_cost_per_step == pytest.approx(0.01)
        assert stats[0].avg_latency_per_step == pytest.approx(500)

    def test_multiple_models_sorted_by_cost(self):
        reporter = Reporter(_make_store())
        steps = [
            _step(model="cheap", actual=_metrics(cost=0.001)),
            _step(model="expensive", actual=_metrics(cost=0.10)),
            _step(model="cheap", actual=_metrics(cost=0.002)),
        ]
        stats = reporter.model_stats(steps)
        assert len(stats) == 2
        # Sorted by cost descending
        assert stats[0].model == "expensive"
        assert stats[1].model == "cheap"
        assert stats[1].step_count == 2
        assert stats[1].total_cost_usd == pytest.approx(0.003)

    def test_step_without_actual_counted_but_no_metrics(self):
        reporter = Reporter(_make_store())
        steps = [_step(model="m", actual=None)]
        stats = reporter.model_stats(steps)
        assert len(stats) == 1
        assert stats[0].step_count == 1
        assert stats[0].total_cost_usd == 0.0

    def test_averages_computed_correctly(self):
        reporter = Reporter(_make_store())
        steps = [
            _step(step_id="s1", model="m", actual=_metrics(cost=0.02, latency=400)),
            _step(step_id="s2", model="m", actual=_metrics(cost=0.04, latency=600)),
        ]
        stats = reporter.model_stats(steps)
        assert stats[0].avg_cost_per_step == pytest.approx(0.03)
        assert stats[0].avg_latency_per_step == pytest.approx(500)


# ===========================================================================
# Reporter.prediction_accuracy
# ===========================================================================


class TestPredictionAccuracy:
    def test_empty_records(self):
        reporter = Reporter(_make_store())
        acc = reporter.prediction_accuracy([])
        assert acc.sample_count == 0

    def test_no_pairs_with_both(self):
        reporter = Reporter(_make_store())
        steps = [
            _step(pred=_metrics(), actual=None),
            _step(pred=None, actual=_metrics()),
        ]
        acc = reporter.prediction_accuracy(steps)
        assert acc.sample_count == 0

    def test_perfect_predictions(self):
        reporter = Reporter(_make_store())
        m = _metrics(pt=100, ct=200, cost=0.01, latency=500)
        steps = [_step(pred=m, actual=m)]
        acc = reporter.prediction_accuracy(steps)
        assert acc.sample_count == 1
        assert acc.prompt_tokens_mae == 0.0
        assert acc.cost_mae == 0.0
        assert acc.cost_bias == 0.0
        assert acc.cost_mape == 0.0

    def test_mae_and_bias(self):
        reporter = Reporter(_make_store())
        # predicted=100, actual=120 → error=20, bias=+20 (under-predicted)
        steps = [
            _step(
                pred=_metrics(pt=100, ct=200, cost=0.010, latency=500),
                actual=_metrics(pt=120, ct=180, cost=0.012, latency=600),
            ),
        ]
        acc = reporter.prediction_accuracy(steps)
        assert acc.sample_count == 1
        assert acc.prompt_tokens_mae == pytest.approx(20)
        assert acc.completion_tokens_mae == pytest.approx(20)
        assert acc.cost_mae == pytest.approx(0.002)
        assert acc.latency_mae == pytest.approx(100)
        # Bias: actual - predicted (positive = under-predicted)
        assert acc.cost_bias == pytest.approx(0.002)
        assert acc.latency_bias == pytest.approx(100)

    def test_mape(self):
        reporter = Reporter(_make_store())
        # predicted cost=0.01, actual=0.012 → APE = |0.002|/0.01 = 0.2
        steps = [
            _step(
                pred=_metrics(cost=0.01, latency=500),
                actual=_metrics(cost=0.012, latency=600),
            ),
        ]
        acc = reporter.prediction_accuracy(steps)
        assert acc.cost_mape == pytest.approx(0.2)
        assert acc.latency_mape == pytest.approx(0.2)

    def test_mape_with_zero_predicted(self):
        reporter = Reporter(_make_store())
        # Zero predicted cost → skip in MAPE calculation
        steps = [
            _step(
                pred=_metrics(cost=0.0, latency=0.0),
                actual=_metrics(cost=0.01, latency=500),
            ),
        ]
        acc = reporter.prediction_accuracy(steps)
        assert acc.cost_mape == 0.0  # no valid samples
        assert acc.latency_mape == 0.0

    def test_multiple_samples_averaged(self):
        reporter = Reporter(_make_store())
        steps = [
            _step(
                step_id="s1",
                pred=_metrics(pt=100, cost=0.01),
                actual=_metrics(pt=110, cost=0.015),
            ),
            _step(
                step_id="s2",
                pred=_metrics(pt=100, cost=0.01),
                actual=_metrics(pt=90, cost=0.008),
            ),
        ]
        acc = reporter.prediction_accuracy(steps)
        assert acc.sample_count == 2
        # MAE: (10 + 10) / 2 = 10
        assert acc.prompt_tokens_mae == pytest.approx(10)
        # Cost MAE: (0.005 + 0.002) / 2 = 0.0035
        assert acc.cost_mae == pytest.approx(0.0035)
        # Cost bias: (0.005 + (-0.002)) / 2 = 0.0015
        assert acc.cost_bias == pytest.approx(0.0015)


# ===========================================================================
# Reporter.budget_compliance
# ===========================================================================


class TestBudgetCompliance:
    def test_empty_run_list(self):
        store = _make_store()
        reporter = Reporter(store)
        c = reporter.budget_compliance([])
        assert c.total_runs == 0
        assert c.compliance_rate == 1.0

    def test_all_within_budget(self):
        store = _make_store()
        store.log_run(_run(run_id="r1", cost=0.05, success=True))
        store.log_run(_run(run_id="r2", cost=0.03, success=True))
        reporter = Reporter(store)
        c = reporter.budget_compliance(["r1", "r2"], budget_caps={"usd": 0.10})
        assert c.total_runs == 2
        assert c.runs_within_budget == 2
        assert c.runs_exceeded == 0
        assert c.compliance_rate == 1.0

    def test_cost_exceeded(self):
        store = _make_store()
        store.log_run(_run(run_id="r1", cost=0.15, success=True))
        reporter = Reporter(store)
        c = reporter.budget_compliance(["r1"], budget_caps={"usd": 0.10})
        assert c.runs_exceeded == 1
        assert c.compliance_rate == 0.0

    def test_token_exceeded(self):
        store = _make_store()
        store.log_run(_run(run_id="r1", tokens=2000, success=True))
        reporter = Reporter(store)
        c = reporter.budget_compliance(["r1"], budget_caps={"tokens": 1000})
        assert c.runs_exceeded == 1

    def test_failed_run_counts_as_exceeded(self):
        store = _make_store()
        store.log_run(_run(run_id="r1", cost=0.01, success=False))
        reporter = Reporter(store)
        c = reporter.budget_compliance(["r1"])
        assert c.runs_exceeded == 1
        assert c.compliance_rate == 0.0

    def test_missing_run_id_skipped(self):
        store = _make_store()
        store.log_run(_run(run_id="r1", success=True))
        reporter = Reporter(store)
        c = reporter.budget_compliance(["r1", "nonexistent"])
        assert c.total_runs == 1

    def test_no_caps_successful_runs_within(self):
        store = _make_store()
        store.log_run(_run(run_id="r1", success=True))
        store.log_run(_run(run_id="r2", success=None))
        reporter = Reporter(store)
        c = reporter.budget_compliance(["r1", "r2"])
        assert c.runs_within_budget == 2
        assert c.compliance_rate == 1.0

    def test_mixed_compliance(self):
        store = _make_store()
        store.log_run(_run(run_id="r1", cost=0.05, success=True))
        store.log_run(_run(run_id="r2", cost=0.20, success=True))
        store.log_run(_run(run_id="r3", cost=0.08, success=False))
        reporter = Reporter(store)
        c = reporter.budget_compliance(
            ["r1", "r2", "r3"], budget_caps={"usd": 0.10}
        )
        assert c.total_runs == 3
        assert c.runs_within_budget == 1
        assert c.runs_exceeded == 2
        assert c.compliance_rate == pytest.approx(1 / 3)


# ===========================================================================
# Reporter.full_report
# ===========================================================================


class TestFullReport:
    def test_empty(self):
        store = _make_store()
        reporter = Reporter(store)
        report = reporter.full_report([])
        assert report.run_summaries == []
        assert report.model_stats == []
        assert report.prediction_accuracy.sample_count == 0
        assert report.budget_compliance.total_runs == 0

    def test_integrates_all_sections(self):
        store = _make_store()
        store.log_run(_run(run_id="r1", cost=0.05, tokens=500, steps=2, success=True))
        store.log_step(_step(
            run_id="r1", step_id="s1", model="gpt-4o",
            pred=_metrics(pt=100, cost=0.02),
            actual=_metrics(pt=120, cost=0.025),
        ))
        store.log_step(_step(
            run_id="r1", step_id="s2", model="gpt-4o",
            pred=_metrics(pt=100, cost=0.02),
            actual=_metrics(pt=110, cost=0.025),
        ))
        reporter = Reporter(store)
        report = reporter.full_report(["r1"], budget_caps={"usd": 0.10})

        # Run summaries
        assert len(report.run_summaries) == 1
        assert report.run_summaries[0].run_id == "r1"

        # Model stats
        assert len(report.model_stats) == 1
        assert report.model_stats[0].model == "gpt-4o"
        assert report.model_stats[0].step_count == 2

        # Prediction accuracy
        assert report.prediction_accuracy.sample_count == 2

        # Budget compliance
        assert report.budget_compliance.total_runs == 1
        assert report.budget_compliance.compliance_rate == 1.0

    def test_multiple_runs(self):
        store = _make_store()
        store.log_run(_run(run_id="r1", cost=0.05, success=True))
        store.log_run(_run(run_id="r2", cost=0.03, success=True))
        store.log_step(_step(run_id="r1", step_id="s1", model="a",
                             actual=_metrics(cost=0.05)))
        store.log_step(_step(run_id="r2", step_id="s1", model="b",
                             actual=_metrics(cost=0.03)))
        reporter = Reporter(store)
        report = reporter.full_report(["r1", "r2"])
        assert len(report.run_summaries) == 2
        assert len(report.model_stats) == 2


# ===========================================================================
# Dataclass defaults
# ===========================================================================


class TestDataclassDefaults:
    def test_model_stats_defaults(self):
        s = ModelStats(model="m")
        assert s.step_count == 0
        assert s.avg_cost_per_step == 0.0

    def test_prediction_accuracy_defaults(self):
        a = PredictionAccuracy()
        assert a.sample_count == 0
        assert a.cost_mape == 0.0

    def test_compliance_defaults(self):
        c = BudgetComplianceReport()
        assert c.compliance_rate == 1.0

    def test_full_report_defaults(self):
        r = FullReport()
        assert r.run_summaries == []
        assert r.model_stats == []
