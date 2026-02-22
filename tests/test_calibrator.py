"""Tests for budgeteer.calibrator — prediction calibration and correction factors."""

import pytest

from budgeteer.calibrator import Calibrator, CorrectionFactors, PredictionError
from budgeteer.models import StepDecision, StepMetrics, StepRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pred(pt=100, ct=200, cost=0.01, latency=500.0):
    return StepMetrics(
        prompt_tokens=pt,
        completion_tokens=ct,
        cost_usd=cost,
        latency_ms=latency,
    )


def _actual(pt=120, ct=180, cost=0.012, latency=600.0):
    return StepMetrics(
        prompt_tokens=pt,
        completion_tokens=ct,
        cost_usd=cost,
        latency_ms=latency,
    )


def _step_record(model="gpt-4o", predicted=None, actual=None):
    return StepRecord(
        run_id="r1",
        step_id="s1",
        decision=StepDecision(model=model),
        predicted=predicted,
        actual=actual,
    )


# ===========================================================================
# CorrectionFactors defaults
# ===========================================================================


class TestCorrectionFactors:
    def test_defaults_are_unity(self):
        f = CorrectionFactors()
        assert f.prompt_tokens == 1.0
        assert f.completion_tokens == 1.0
        assert f.cost_usd == 1.0
        assert f.latency_ms == 1.0
        assert f.sample_count == 0


# ===========================================================================
# PredictionError
# ===========================================================================


class TestPredictionError:
    def test_defaults(self):
        e = PredictionError()
        assert e.prompt_tokens_ratio is None

    def test_custom(self):
        e = PredictionError(cost_ratio=1.5)
        assert e.cost_ratio == 1.5


# ===========================================================================
# Calibrator basics
# ===========================================================================


class TestCalibratorBasics:
    def test_invalid_alpha(self):
        with pytest.raises(ValueError):
            Calibrator(alpha=0.0)
        with pytest.raises(ValueError):
            Calibrator(alpha=-0.1)
        with pytest.raises(ValueError):
            Calibrator(alpha=1.5)

    def test_alpha_1_is_valid(self):
        cal = Calibrator(alpha=1.0)
        assert cal is not None

    def test_no_models_initially(self):
        cal = Calibrator()
        assert cal.models == []

    def test_unseen_model_returns_default_factors(self):
        cal = Calibrator()
        f = cal.get_factors("unknown")
        assert f.prompt_tokens == 1.0
        assert f.sample_count == 0


# ===========================================================================
# Single update
# ===========================================================================


class TestSingleUpdate:
    def test_update_creates_model_entry(self):
        cal = Calibrator(alpha=0.3)
        cal.update("gpt-4o", _pred(), _actual())
        assert "gpt-4o" in cal.models

    def test_update_increments_sample_count(self):
        cal = Calibrator()
        cal.update("gpt-4o", _pred(), _actual())
        assert cal.get_factors("gpt-4o").sample_count == 1

    def test_returns_prediction_error(self):
        cal = Calibrator()
        err = cal.update("gpt-4o", _pred(pt=100), _actual(pt=120))
        # ratio = 120/100 = 1.2
        assert err.prompt_tokens_ratio == pytest.approx(1.2)

    def test_perfect_prediction_ratio_is_one(self):
        cal = Calibrator()
        m = _pred(pt=100, ct=200, cost=0.01, latency=500)
        err = cal.update("m", m, m)
        assert err.prompt_tokens_ratio == pytest.approx(1.0)
        assert err.cost_ratio == pytest.approx(1.0)

    def test_zero_predicted_gives_none_ratio(self):
        cal = Calibrator()
        err = cal.update("m", _pred(pt=0, ct=0, cost=0.0, latency=0.0), _actual())
        assert err.prompt_tokens_ratio is None
        assert err.cost_ratio is None
        # Factors should remain at default (1.0) since ratio is None
        f = cal.get_factors("m")
        assert f.prompt_tokens == 1.0


# ===========================================================================
# EMA convergence
# ===========================================================================


class TestEMAConvergence:
    def test_factors_move_toward_ratio(self):
        cal = Calibrator(alpha=0.3)
        # predicted=100, actual=200 → ratio=2.0
        # After one update: factor = 1.0 * 0.7 + 2.0 * 0.3 = 1.3
        cal.update("m", _pred(pt=100), _actual(pt=200))
        f = cal.get_factors("m")
        assert f.prompt_tokens == pytest.approx(1.3)

    def test_repeated_updates_converge(self):
        cal = Calibrator(alpha=0.3)
        # Repeatedly observe actual=150 when predicting 100 → ratio=1.5
        for _ in range(50):
            cal.update("m", _pred(pt=100), _actual(pt=150))
        f = cal.get_factors("m")
        # Should converge close to 1.5
        assert f.prompt_tokens == pytest.approx(1.5, abs=0.01)

    def test_alpha_1_uses_latest_only(self):
        cal = Calibrator(alpha=1.0)
        cal.update("m", _pred(pt=100), _actual(pt=200))  # ratio=2.0
        cal.update("m", _pred(pt=100), _actual(pt=300))  # ratio=3.0
        f = cal.get_factors("m")
        assert f.prompt_tokens == pytest.approx(3.0)

    def test_per_model_independence(self):
        cal = Calibrator(alpha=0.5)
        cal.update("cheap", _pred(cost=0.01), _actual(cost=0.02))
        cal.update("expensive", _pred(cost=0.10), _actual(cost=0.05))

        f_cheap = cal.get_factors("cheap")
        f_exp = cal.get_factors("expensive")
        # cheap: ratio=2.0, factor = 1.0*0.5 + 2.0*0.5 = 1.5
        assert f_cheap.cost_usd == pytest.approx(1.5)
        # expensive: ratio=0.5, factor = 1.0*0.5 + 0.5*0.5 = 0.75
        assert f_exp.cost_usd == pytest.approx(0.75)


# ===========================================================================
# Apply corrections
# ===========================================================================


class TestApply:
    def test_apply_with_no_data_is_identity(self):
        cal = Calibrator()
        pred = _pred(pt=100, ct=200, cost=0.01, latency=500)
        corrected = cal.apply("unknown", pred)
        assert corrected.prompt_tokens == 100
        assert corrected.completion_tokens == 200
        assert corrected.cost_usd == pytest.approx(0.01)

    def test_apply_scales_predictions(self):
        cal = Calibrator(alpha=1.0)
        # ratio=2.0 for all metrics
        cal.update("m", _pred(pt=100, ct=200, cost=0.01, latency=500),
                       _actual(pt=200, ct=400, cost=0.02, latency=1000))

        corrected = cal.apply("m", _pred(pt=100, ct=200, cost=0.01, latency=500))
        assert corrected.prompt_tokens == 200
        assert corrected.completion_tokens == 400
        assert corrected.cost_usd == pytest.approx(0.02)
        assert corrected.latency_ms == pytest.approx(1000)

    def test_apply_preserves_non_metric_fields(self):
        cal = Calibrator()
        pred = StepMetrics(
            prompt_tokens=100, completion_tokens=200,
            cost_usd=0.01, latency_ms=500,
            tool_calls_made=3, success=True,
        )
        corrected = cal.apply("m", pred)
        assert corrected.tool_calls_made == 3
        assert corrected.success is True

    def test_apply_min_one_token(self):
        cal = Calibrator(alpha=1.0)
        # Very low ratio
        cal.update("m", _pred(pt=100), _actual(pt=1))
        corrected = cal.apply("m", _pred(pt=1))
        assert corrected.prompt_tokens >= 1


# ===========================================================================
# Bulk update and step records
# ===========================================================================


class TestBulkUpdate:
    def test_update_from_record_with_both(self):
        cal = Calibrator()
        rec = _step_record(predicted=_pred(), actual=_actual())
        err = cal.update_from_record(rec)
        assert err is not None
        assert cal.get_factors("gpt-4o").sample_count == 1

    def test_update_from_record_without_predicted(self):
        cal = Calibrator()
        rec = _step_record(predicted=None, actual=_actual())
        assert cal.update_from_record(rec) is None

    def test_update_from_record_without_actual(self):
        cal = Calibrator()
        rec = _step_record(predicted=_pred(), actual=None)
        assert cal.update_from_record(rec) is None

    def test_bulk_update_counts_valid(self):
        cal = Calibrator()
        records = [
            _step_record(predicted=_pred(), actual=_actual()),
            _step_record(predicted=_pred(), actual=None),  # skipped
            _step_record(predicted=_pred(), actual=_actual()),
        ]
        count = cal.bulk_update(records)
        assert count == 2
        assert cal.get_factors("gpt-4o").sample_count == 2


# ===========================================================================
# Reset
# ===========================================================================


class TestReset:
    def test_reset_single_model(self):
        cal = Calibrator()
        cal.update("a", _pred(), _actual())
        cal.update("b", _pred(), _actual())
        cal.reset("a")
        assert "a" not in cal.models
        assert "b" in cal.models

    def test_reset_all(self):
        cal = Calibrator()
        cal.update("a", _pred(), _actual())
        cal.update("b", _pred(), _actual())
        cal.reset()
        assert cal.models == []

    def test_reset_nonexistent_is_safe(self):
        cal = Calibrator()
        cal.reset("nonexistent")  # no error
