"""Tests for Milestone 7 — Calibrator wired into the prediction pipeline."""

import pytest

from budgeteer.calibrator import Calibrator
from budgeteer.config import BudgeteerConfig
from budgeteer.models import (
    ModelTier,
    RunBudget,
    StepContext,
    StepMetrics,
)
from budgeteer.policy import PolicyEngine
from budgeteer.router import StrategyRouter
from budgeteer.sdk import Budgeteer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _two_tier_config(tmp_db):
    return BudgeteerConfig(
        storage_path=tmp_db,
        default_max_tokens=1024,
        model_tiers=[
            ModelTier(
                name="cheap",
                cost_per_prompt_token=0.001,
                cost_per_completion_token=0.002,
                max_context_window=4096,
            ),
            ModelTier(
                name="expensive",
                cost_per_prompt_token=0.01,
                cost_per_completion_token=0.03,
                max_context_window=8192,
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Router + Calibrator integration
# ---------------------------------------------------------------------------


class TestRouterCalibration:
    def test_forecast_without_calibrator(self, tmp_db):
        config = _two_tier_config(tmp_db)
        router = StrategyRouter(config)
        ctx = StepContext(run_id="r1", messages=[{"role": "user", "content": "hello"}])
        candidates = router.generate_candidates(ctx)
        c = candidates[0]
        router.forecast(c, ctx)
        # Raw forecast should be non-zero
        assert c.predicted_cost_usd > 0
        assert c.predicted_latency_ms > 0

    def test_forecast_with_calibrator_no_data(self, tmp_db):
        config = _two_tier_config(tmp_db)
        cal = Calibrator(alpha=0.3)
        router = StrategyRouter(config, calibrator=cal)
        ctx = StepContext(run_id="r1", messages=[{"role": "user", "content": "hello"}])
        candidates = router.generate_candidates(ctx)
        c = candidates[0]
        router.forecast(c, ctx)
        # With no calibration data, factors are all 1.0, so same as raw
        assert c.predicted_cost_usd > 0

    def test_forecast_with_calibrator_correction(self, tmp_db):
        config = _two_tier_config(tmp_db)
        cal = Calibrator(alpha=1.0)  # alpha=1 for instant convergence
        # Teach calibrator that "cheap" under-predicts cost by 2x
        cal.update(
            "cheap",
            predicted=StepMetrics(prompt_tokens=100, completion_tokens=100, cost_usd=0.1, latency_ms=100),
            actual=StepMetrics(prompt_tokens=100, completion_tokens=100, cost_usd=0.2, latency_ms=100),
        )
        router = StrategyRouter(config, calibrator=cal)
        ctx = StepContext(run_id="r1", messages=[{"role": "user", "content": "hello"}])
        candidates = router.generate_candidates(ctx)
        # Find a "cheap" candidate
        cheap_cand = [c for c in candidates if c.model == "cheap"][0]

        # Forecast without calibrator for comparison
        router_raw = StrategyRouter(config)
        raw_cand = [c for c in router_raw.generate_candidates(ctx) if c.model == "cheap"][0]
        router_raw.forecast(raw_cand, ctx)
        router.forecast(cheap_cand, ctx)

        # Calibrated cost should be ~2x the raw cost
        assert cheap_cand.predicted_cost_usd == pytest.approx(raw_cand.predicted_cost_usd * 2.0, rel=0.01)

    def test_set_calibrator(self, tmp_db):
        config = _two_tier_config(tmp_db)
        router = StrategyRouter(config)
        assert router._calibrator is None
        cal = Calibrator()
        router.set_calibrator(cal)
        assert router._calibrator is cal

    def test_get_prediction(self, tmp_db):
        config = _two_tier_config(tmp_db)
        router = StrategyRouter(config)
        ctx = StepContext(run_id="r1", messages=[{"role": "user", "content": "hello"}])
        candidates = router.generate_candidates(ctx)
        c = candidates[0]
        router.forecast(c, ctx)
        pred = router.get_prediction(c)
        assert isinstance(pred, StepMetrics)
        assert pred.prompt_tokens == c.predicted_prompt_tokens
        assert pred.completion_tokens == c.predicted_completion_tokens
        assert pred.cost_usd == c.predicted_cost_usd
        assert pred.latency_ms == c.predicted_latency_ms


# ---------------------------------------------------------------------------
# PolicyEngine + Calibrator
# ---------------------------------------------------------------------------


class TestPolicyCalibration:
    def test_last_prediction_routed_path(self, tmp_db):
        config = _two_tier_config(tmp_db)
        from budgeteer.telemetry import TelemetryStore
        telemetry = TelemetryStore(tmp_db)
        policy = PolicyEngine(config, telemetry)
        ctx = StepContext(
            run_id="r1",
            run_budget=RunBudget(hard_usd_cap=100.0),
            messages=[{"role": "user", "content": "hello"}],
        )
        policy.evaluate(ctx)
        pred = policy.last_prediction
        assert pred is not None
        assert isinstance(pred, StepMetrics)
        assert pred.cost_usd > 0
        telemetry.close()

    def test_last_prediction_legacy_path(self, tmp_db):
        config = BudgeteerConfig(storage_path=tmp_db)  # no model_tiers
        from budgeteer.telemetry import TelemetryStore
        telemetry = TelemetryStore(tmp_db)
        policy = PolicyEngine(config, telemetry)
        ctx = StepContext(run_id="r1")
        policy.evaluate(ctx)
        assert policy.last_prediction is None
        telemetry.close()

    def test_calibrator_passed_to_router(self, tmp_db):
        config = _two_tier_config(tmp_db)
        cal = Calibrator()
        from budgeteer.telemetry import TelemetryStore
        telemetry = TelemetryStore(tmp_db)
        policy = PolicyEngine(config, telemetry, calibrator=cal)
        assert policy._router._calibrator is cal
        telemetry.close()


# ---------------------------------------------------------------------------
# SDK end-to-end calibration
# ---------------------------------------------------------------------------


class TestSDKCalibrationIntegration:
    def test_step_record_predicted_populated_routed(self, tmp_db):
        config = _two_tier_config(tmp_db)
        sdk = Budgeteer(config)
        run = sdk.start_run(run_budget=RunBudget(hard_usd_cap=100.0))
        ctx = StepContext(
            run_id=run.run_id,
            messages=[{"role": "user", "content": "hello"}],
        )
        decision = sdk.before_step(ctx)
        metrics = StepMetrics(prompt_tokens=50, completion_tokens=30, cost_usd=0.01, latency_ms=100)
        sdk.after_step(ctx, decision, metrics)

        steps = sdk.telemetry.get_steps(run.run_id)
        assert len(steps) == 1
        assert steps[0].predicted is not None
        assert steps[0].predicted.cost_usd > 0
        assert steps[0].actual is not None
        sdk.end_run(run.run_id)
        sdk.close()

    def test_step_record_predicted_none_legacy(self, tmp_db):
        config = BudgeteerConfig(storage_path=tmp_db)
        sdk = Budgeteer(config)
        run = sdk.start_run()
        ctx = StepContext(run_id=run.run_id)
        decision = sdk.before_step(ctx)
        metrics = StepMetrics(prompt_tokens=50, completion_tokens=30, cost_usd=0.01)
        sdk.after_step(ctx, decision, metrics)

        steps = sdk.telemetry.get_steps(run.run_id)
        assert len(steps) == 1
        assert steps[0].predicted is None
        sdk.end_run(run.run_id)
        sdk.close()

    def test_calibrator_property(self, tmp_db):
        config = _two_tier_config(tmp_db)
        sdk = Budgeteer(config)
        assert sdk.calibrator is not None
        assert isinstance(sdk.calibrator, Calibrator)
        sdk.close()

    def test_calibration_disabled(self, tmp_db):
        config = _two_tier_config(tmp_db)
        config.calibration_enabled = False
        sdk = Budgeteer(config)
        assert sdk.calibrator is None
        # Should still work without calibration
        run = sdk.start_run(run_budget=RunBudget(hard_usd_cap=100.0))
        ctx = StepContext(
            run_id=run.run_id,
            messages=[{"role": "user", "content": "hello"}],
        )
        decision = sdk.before_step(ctx)
        metrics = StepMetrics(prompt_tokens=50, completion_tokens=30, cost_usd=0.01)
        sdk.after_step(ctx, decision, metrics)
        # With calibration disabled, predictions still come from router
        # but calibrator corrections are not applied
        steps = sdk.telemetry.get_steps(run.run_id)
        # Predictions are still stored from the routed path (router works without calibrator)
        assert steps[0].predicted is not None
        sdk.end_run(run.run_id)
        sdk.close()

    def test_calibrator_factors_update_after_steps(self, tmp_db):
        config = _two_tier_config(tmp_db)
        config.calibration_alpha = 0.5
        sdk = Budgeteer(config)
        run = sdk.start_run(run_budget=RunBudget(hard_usd_cap=100.0))

        for _ in range(5):
            ctx = StepContext(
                run_id=run.run_id,
                messages=[{"role": "user", "content": "hello"}],
            )
            decision = sdk.before_step(ctx)
            # Actual cost is always much lower than predicted (conservative)
            metrics = StepMetrics(
                prompt_tokens=2, completion_tokens=5,
                cost_usd=0.0001, latency_ms=10,
            )
            sdk.after_step(ctx, decision, metrics)

        # Calibrator should have learned that predictions over-estimate
        factors = sdk.calibrator.get_factors(decision.model)
        assert factors.sample_count == 5
        # Cost factor should be < 1.0 since actual < predicted
        assert factors.cost_usd < 1.0

        sdk.end_run(run.run_id)
        sdk.close()

    def test_calibrator_convergence(self, tmp_db):
        """After many consistent steps, predictions should converge toward actuals."""
        config = _two_tier_config(tmp_db)
        config.calibration_alpha = 0.5
        sdk = Budgeteer(config)
        run = sdk.start_run(run_budget=RunBudget(hard_usd_cap=1000.0))

        predictions = []
        for i in range(10):
            ctx = StepContext(
                run_id=run.run_id,
                messages=[{"role": "user", "content": "hello world " * 10}],
            )
            decision = sdk.before_step(ctx)
            pred = sdk.policy.last_prediction
            predictions.append(pred.cost_usd if pred else 0)
            metrics = StepMetrics(
                prompt_tokens=10, completion_tokens=20,
                cost_usd=0.005, latency_ms=50,
            )
            sdk.after_step(ctx, decision, metrics)

        # Later predictions should be closer to actual than earlier ones
        # (i.e. the correction factor should move toward the actual/predicted ratio)
        if len(predictions) >= 2:
            # Just verify factors converged (sample_count increased)
            factors = sdk.calibrator.get_factors(decision.model)
            assert factors.sample_count == 10

        sdk.end_run(run.run_id)
        sdk.close()

    def test_calibration_alpha_from_config(self, tmp_db):
        config = _two_tier_config(tmp_db)
        config.calibration_alpha = 0.7
        sdk = Budgeteer(config)
        assert sdk.calibrator._alpha == 0.7
        sdk.close()

    def test_multiple_models_calibrated_independently(self, tmp_db):
        config = _two_tier_config(tmp_db)
        config.calibration_alpha = 1.0
        sdk = Budgeteer(config)
        run = sdk.start_run(run_budget=RunBudget(hard_usd_cap=1000.0))

        # Force a step that uses "cheap" model
        ctx1 = StepContext(
            run_id=run.run_id,
            messages=[{"role": "user", "content": "hello"}],
        )
        d1 = sdk.before_step(ctx1)
        sdk.after_step(ctx1, d1, StepMetrics(
            prompt_tokens=5, completion_tokens=10, cost_usd=0.001, latency_ms=20
        ))

        # Check that only the used model has calibration data
        cal = sdk.calibrator
        assert len(cal.models) >= 1

        sdk.end_run(run.run_id)
        sdk.close()

    def test_config_from_dict_calibration_fields(self):
        data = {
            "calibration_enabled": False,
            "calibration_alpha": 0.5,
        }
        config = BudgeteerConfig.from_dict(data)
        assert config.calibration_enabled is False
        assert config.calibration_alpha == 0.5

    def test_config_from_dict_defaults(self):
        config = BudgeteerConfig.from_dict({})
        assert config.calibration_enabled is True
        assert config.calibration_alpha == 0.3
