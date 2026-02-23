"""Tests for enhanced reporting: per-model accuracy & degradation impact (Milestone 16)."""

from __future__ import annotations

import pytest

from budgeteer.models import StepDecision, StepMetrics, StepRecord
from budgeteer.reporting import FullReport, PredictionAccuracy, Reporter
from budgeteer.telemetry import TelemetryStore
from budgeteer.models import RunRecord


def _step(model, degrade_level=0, pred_cost=0.01, actual_cost=0.012,
          pred_latency=100, actual_latency=120, pred_tokens=100, actual_tokens=110):
    """Helper to create a StepRecord with predicted and actual."""
    return StepRecord(
        run_id="r1",
        step_id=f"s-{id(model)}-{degrade_level}",
        decision=StepDecision(model=model, degrade_level=degrade_level),
        predicted=StepMetrics(prompt_tokens=pred_tokens, cost_usd=pred_cost, latency_ms=pred_latency),
        actual=StepMetrics(prompt_tokens=actual_tokens, cost_usd=actual_cost, latency_ms=actual_latency),
    )


class TestPerModelAccuracy:
    """Tests for Reporter.per_model_accuracy()."""

    def test_single_model(self, tmp_path):
        store = TelemetryStore(str(tmp_path / "t.db"))
        reporter = Reporter(store)
        steps = [_step("gpt-4"), _step("gpt-4")]
        result = reporter.per_model_accuracy(steps)
        assert "gpt-4" in result
        assert result["gpt-4"].sample_count == 2
        store.close()

    def test_multiple_models(self, tmp_path):
        store = TelemetryStore(str(tmp_path / "t.db"))
        reporter = Reporter(store)
        steps = [
            _step("gpt-4", pred_cost=0.01, actual_cost=0.015),
            _step("gpt-3.5", pred_cost=0.001, actual_cost=0.0012),
            _step("gpt-4", pred_cost=0.02, actual_cost=0.018),
        ]
        result = reporter.per_model_accuracy(steps)
        assert len(result) == 2
        assert result["gpt-4"].sample_count == 2
        assert result["gpt-3.5"].sample_count == 1
        store.close()

    def test_accuracy_values_are_reasonable(self, tmp_path):
        store = TelemetryStore(str(tmp_path / "t.db"))
        reporter = Reporter(store)
        steps = [_step("m", pred_cost=0.01, actual_cost=0.012)]
        result = reporter.per_model_accuracy(steps)
        acc = result["m"]
        assert acc.cost_mae == pytest.approx(0.002)
        assert acc.cost_bias == pytest.approx(0.002)  # actual > predicted
        store.close()

    def test_empty_steps(self, tmp_path):
        store = TelemetryStore(str(tmp_path / "t.db"))
        reporter = Reporter(store)
        result = reporter.per_model_accuracy([])
        assert result == {}
        store.close()

    def test_steps_without_predictions(self, tmp_path):
        store = TelemetryStore(str(tmp_path / "t.db"))
        reporter = Reporter(store)
        step = StepRecord(
            run_id="r1", step_id="s1",
            decision=StepDecision(model="m"),
            predicted=None,
            actual=StepMetrics(cost_usd=0.01),
        )
        result = reporter.per_model_accuracy([step])
        assert result["m"].sample_count == 0
        store.close()


class TestDegradationImpact:
    """Tests for Reporter.degradation_impact()."""

    def test_groups_by_level(self, tmp_path):
        store = TelemetryStore(str(tmp_path / "t.db"))
        reporter = Reporter(store)
        steps = [
            _step("m", degrade_level=0, actual_cost=0.02, actual_latency=100),
            _step("m", degrade_level=0, actual_cost=0.03, actual_latency=150),
            _step("m", degrade_level=1, actual_cost=0.01, actual_latency=80),
        ]
        result = reporter.degradation_impact(steps)
        assert len(result) == 2
        assert result[0]["degrade_level"] == 0
        assert result[0]["step_count"] == 2
        assert result[1]["degrade_level"] == 1
        assert result[1]["step_count"] == 1
        store.close()

    def test_avg_cost_computed(self, tmp_path):
        store = TelemetryStore(str(tmp_path / "t.db"))
        reporter = Reporter(store)
        steps = [
            _step("m", degrade_level=0, actual_cost=0.02, actual_latency=100),
            _step("m", degrade_level=0, actual_cost=0.04, actual_latency=200),
        ]
        result = reporter.degradation_impact(steps)
        assert result[0]["avg_cost_usd"] == pytest.approx(0.03)
        assert result[0]["avg_latency_ms"] == pytest.approx(150.0)
        store.close()

    def test_empty_steps(self, tmp_path):
        store = TelemetryStore(str(tmp_path / "t.db"))
        reporter = Reporter(store)
        result = reporter.degradation_impact([])
        assert result == []
        store.close()

    def test_sorted_by_level(self, tmp_path):
        store = TelemetryStore(str(tmp_path / "t.db"))
        reporter = Reporter(store)
        steps = [
            _step("m", degrade_level=2),
            _step("m", degrade_level=0),
            _step("m", degrade_level=1),
        ]
        result = reporter.degradation_impact(steps)
        levels = [r["degrade_level"] for r in result]
        assert levels == [0, 1, 2]
        store.close()

    def test_steps_without_actuals(self, tmp_path):
        store = TelemetryStore(str(tmp_path / "t.db"))
        reporter = Reporter(store)
        step = StepRecord(
            run_id="r1", step_id="s1",
            decision=StepDecision(model="m", degrade_level=0),
            actual=None,
        )
        result = reporter.degradation_impact([step])
        assert result[0]["avg_cost_usd"] == 0.0
        store.close()


class TestFullReportIncludes:
    """Full report includes new fields."""

    def test_full_report_has_per_model_accuracy(self, tmp_path):
        store = TelemetryStore(str(tmp_path / "t.db"))
        store.log_run(RunRecord(run_id="r1"))
        store.log_step(StepRecord(
            run_id="r1", step_id="s1",
            decision=StepDecision(model="gpt-4"),
            predicted=StepMetrics(cost_usd=0.01),
            actual=StepMetrics(cost_usd=0.012),
        ))
        reporter = Reporter(store)
        report = reporter.full_report(["r1"])
        assert report.per_model_accuracy is not None
        assert "gpt-4" in report.per_model_accuracy
        store.close()

    def test_full_report_has_degradation_impact(self, tmp_path):
        store = TelemetryStore(str(tmp_path / "t.db"))
        store.log_run(RunRecord(run_id="r1"))
        store.log_step(StepRecord(
            run_id="r1", step_id="s1",
            decision=StepDecision(model="m", degrade_level=0),
            actual=StepMetrics(cost_usd=0.01),
        ))
        reporter = Reporter(store)
        report = reporter.full_report(["r1"])
        assert report.degradation_impact is not None
        assert len(report.degradation_impact) >= 1
        store.close()

    def test_full_report_empty_runs(self, tmp_path):
        store = TelemetryStore(str(tmp_path / "t.db"))
        reporter = Reporter(store)
        report = reporter.full_report([])
        assert report.per_model_accuracy == {}
        assert report.degradation_impact == []
        store.close()
