"""Tests for configurable degradation ladder and thresholds (Milestone 15)."""

from __future__ import annotations

import pytest

from budgeteer.config import BudgeteerConfig
from budgeteer.models import ModelTier, RunBudget, StepContext
from budgeteer.policy import PolicyEngine
from budgeteer.router import DegradeLevelParams, StrategyRouter
from budgeteer.telemetry import TelemetryStore


def _make_tier(name="m", prompt=0.001, completion=0.002, ctx=8192):
    return ModelTier(name=name, cost_per_prompt_token=prompt, cost_per_completion_token=completion, max_context_window=ctx)


class TestCustomDegradeLadder:
    """Tests for custom degradation ladders in the router."""

    def test_default_ladder_has_5_levels(self, tmp_path):
        cfg = BudgeteerConfig(model_tiers=[_make_tier()], storage_path=str(tmp_path / "t.db"))
        router = StrategyRouter(cfg)
        ctx = StepContext(run_id="r1")
        candidates = router.generate_candidates(ctx)
        # 1 tier * 5 levels
        assert len(candidates) == 5

    def test_custom_3_level_ladder(self, tmp_path):
        ladder = [
            {"max_tokens_ratio": 1.0, "tool_calls": 5, "retrieval": True, "retrieval_top_k": 3, "quality": 1.0},
            {"max_tokens_ratio": 0.5, "tool_calls": 2, "retrieval": True, "retrieval_top_k": 1, "quality": 0.7},
            {"max_tokens_ratio": 0.25, "tool_calls": 0, "retrieval": False, "retrieval_top_k": 0, "quality": 0.3},
        ]
        cfg = BudgeteerConfig(model_tiers=[_make_tier()], degrade_ladder=ladder, storage_path=str(tmp_path / "t.db"))
        router = StrategyRouter(cfg)
        ctx = StepContext(run_id="r1")
        candidates = router.generate_candidates(ctx)
        assert len(candidates) == 3  # 1 tier * 3 levels

    def test_custom_ladder_quality_scores(self, tmp_path):
        ladder = [
            {"max_tokens_ratio": 1.0, "quality": 0.9},
            {"max_tokens_ratio": 0.5, "quality": 0.4},
        ]
        cfg = BudgeteerConfig(model_tiers=[_make_tier()], degrade_ladder=ladder, storage_path=str(tmp_path / "t.db"))
        router = StrategyRouter(cfg)
        ctx = StepContext(run_id="r1")
        candidates = router.generate_candidates(ctx)
        # Level 0 should have higher quality than level 1
        assert candidates[0].quality_score > candidates[1].quality_score

    def test_custom_ladder_max_tokens_ratio_applied(self, tmp_path):
        ladder = [
            {"max_tokens_ratio": 1.0, "quality": 1.0},
            {"max_tokens_ratio": 0.3, "quality": 0.5},
        ]
        cfg = BudgeteerConfig(
            model_tiers=[_make_tier()],
            degrade_ladder=ladder,
            default_max_tokens=1000,
            storage_path=str(tmp_path / "t.db"),
        )
        router = StrategyRouter(cfg)
        ctx = StepContext(run_id="r1")
        candidates = router.generate_candidates(ctx)
        assert candidates[0].max_tokens == 1000
        assert candidates[1].max_tokens == 300

    def test_custom_ladder_tool_calls(self, tmp_path):
        ladder = [
            {"max_tokens_ratio": 1.0, "tool_calls": 10, "quality": 1.0},
        ]
        cfg = BudgeteerConfig(model_tiers=[_make_tier()], degrade_ladder=ladder, storage_path=str(tmp_path / "t.db"))
        router = StrategyRouter(cfg)
        ctx = StepContext(run_id="r1")
        candidates = router.generate_candidates(ctx)
        assert candidates[0].tool_calls_allowed == 10

    def test_custom_ladder_retrieval_disabled(self, tmp_path):
        ladder = [
            {"max_tokens_ratio": 1.0, "retrieval": False, "retrieval_top_k": 0, "quality": 1.0},
        ]
        cfg = BudgeteerConfig(model_tiers=[_make_tier()], degrade_ladder=ladder, storage_path=str(tmp_path / "t.db"))
        router = StrategyRouter(cfg)
        ctx = StepContext(run_id="r1")
        candidates = router.generate_candidates(ctx)
        assert candidates[0].retrieval_enabled is False

    def test_multiple_tiers_with_custom_ladder(self, tmp_path):
        tiers = [_make_tier("cheap", 0.001, 0.002), _make_tier("expensive", 0.01, 0.03)]
        ladder = [
            {"max_tokens_ratio": 1.0, "quality": 1.0},
            {"max_tokens_ratio": 0.5, "quality": 0.5},
        ]
        cfg = BudgeteerConfig(model_tiers=tiers, degrade_ladder=ladder, storage_path=str(tmp_path / "t.db"))
        router = StrategyRouter(cfg)
        ctx = StepContext(run_id="r1")
        candidates = router.generate_candidates(ctx)
        assert len(candidates) == 4  # 2 tiers * 2 levels


class TestCustomDegradeLadderValidation:
    """Validation of custom ladders."""

    def test_empty_ladder_rejected(self, tmp_path):
        cfg = BudgeteerConfig(model_tiers=[_make_tier()], degrade_ladder=[], storage_path=str(tmp_path / "t.db"))
        with pytest.raises(ValueError, match="at least 1 level"):
            StrategyRouter(cfg)

    def test_invalid_max_tokens_ratio_above_1(self, tmp_path):
        ladder = [{"max_tokens_ratio": 1.5, "quality": 1.0}]
        cfg = BudgeteerConfig(model_tiers=[_make_tier()], degrade_ladder=ladder, storage_path=str(tmp_path / "t.db"))
        with pytest.raises(ValueError, match="max_tokens_ratio"):
            StrategyRouter(cfg)

    def test_invalid_quality_negative(self, tmp_path):
        ladder = [{"max_tokens_ratio": 1.0, "quality": -0.1}]
        cfg = BudgeteerConfig(model_tiers=[_make_tier()], degrade_ladder=ladder, storage_path=str(tmp_path / "t.db"))
        with pytest.raises(ValueError, match="quality"):
            StrategyRouter(cfg)

    def test_invalid_quality_above_1(self, tmp_path):
        ladder = [{"max_tokens_ratio": 1.0, "quality": 1.5}]
        cfg = BudgeteerConfig(model_tiers=[_make_tier()], degrade_ladder=ladder, storage_path=str(tmp_path / "t.db"))
        with pytest.raises(ValueError, match="quality"):
            StrategyRouter(cfg)


class TestCustomDegradeThresholds:
    """Tests for configurable degradation thresholds in PolicyEngine (legacy path)."""

    def test_default_thresholds(self, tmp_path):
        cfg = BudgeteerConfig(storage_path=str(tmp_path / "t.db"))
        tel = TelemetryStore(cfg.storage_path)
        pe = PolicyEngine(cfg, tel)
        assert pe._degrade_level_1 == 0.8
        assert pe._degrade_level_2 == 0.9
        tel.close()

    def test_custom_thresholds(self, tmp_path):
        cfg = BudgeteerConfig(
            degrade_thresholds=(0.5, 0.7),
            storage_path=str(tmp_path / "t.db"),
        )
        tel = TelemetryStore(cfg.storage_path)
        pe = PolicyEngine(cfg, tel)
        assert pe._degrade_level_1 == 0.5
        assert pe._degrade_level_2 == 0.7
        tel.close()

    def test_custom_threshold_triggers_earlier(self, tmp_path):
        """With thresholds (0.5, 0.7), 55% usage should trigger level 1."""
        cfg = BudgeteerConfig(
            degrade_thresholds=(0.5, 0.7),
            storage_path=str(tmp_path / "t.db"),
        )
        tel = TelemetryStore(cfg.storage_path)
        pe = PolicyEngine(cfg, tel)
        # 55% consumed -> level 1 with threshold at 0.5
        assert pe._degradation_level(0.55) == 1
        # Same 55% with default (0.8) would be level 0
        tel.close()

    def test_custom_threshold_level_2(self, tmp_path):
        cfg = BudgeteerConfig(
            degrade_thresholds=(0.5, 0.7),
            storage_path=str(tmp_path / "t.db"),
        )
        tel = TelemetryStore(cfg.storage_path)
        pe = PolicyEngine(cfg, tel)
        assert pe._degradation_level(0.75) == 2
        tel.close()

    def test_default_behavior_unchanged(self, tmp_path):
        """Default thresholds should give same behavior as before."""
        cfg = BudgeteerConfig(storage_path=str(tmp_path / "t.db"))
        tel = TelemetryStore(cfg.storage_path)
        pe = PolicyEngine(cfg, tel)
        assert pe._degradation_level(0.5) == 0
        assert pe._degradation_level(0.85) == 1
        assert pe._degradation_level(0.95) == 2
        tel.close()

    def test_custom_thresholds_in_legacy_evaluation(self, tmp_path):
        """Full legacy evaluation with custom thresholds."""
        from budgeteer.models import RunRecord
        from budgeteer.sdk import Budgeteer

        cfg = BudgeteerConfig(
            degrade_thresholds=(0.4, 0.6),
            storage_path=str(tmp_path / "t.db"),
        )
        b = Budgeteer(config=cfg)
        run = b.start_run(run_budget=RunBudget(hard_usd_cap=1.0))
        # 50% consumed -> should hit level 1 with threshold at 0.4
        b._active_runs[run.run_id].total_cost_usd = 0.5
        ctx = StepContext(run_id=run.run_id)
        decision = b.before_step(ctx)
        assert decision.degrade_level >= 1
        b.close()

    def test_no_ladder_uses_default(self, tmp_path):
        """When degrade_ladder is None, default 5-level ladder is used."""
        cfg = BudgeteerConfig(model_tiers=[_make_tier()], storage_path=str(tmp_path / "t.db"))
        router = StrategyRouter(cfg)
        ctx = StepContext(run_id="r1")
        candidates = router.generate_candidates(ctx)
        assert len(candidates) == 5  # default has 5 levels
