"""Tests for budgeteer.router — strategy routing, forecasting, and selection."""

import pytest

from budgeteer.config import BudgeteerConfig
from budgeteer.exceptions import BudgetExceededError
from budgeteer.models import (
    BudgetAccount,
    BudgetScope,
    ModelTier,
    RunBudget,
    StepContext,
)
from budgeteer.policy import PolicyEngine
from budgeteer.router import (
    DEFAULT_DEGRADE_LADDER,
    CandidateStrategy,
    StrategyRouter,
)
from budgeteer.telemetry import TelemetryStore

# ---------------------------------------------------------------------------
# Shared model tiers for tests
# ---------------------------------------------------------------------------

CHEAP = ModelTier(
    name="gpt-4o-mini",
    cost_per_prompt_token=0.00000015,   # $0.15/M
    cost_per_completion_token=0.0000006,  # $0.60/M
    max_context_window=128_000,
    tier="cheap",
)
STANDARD = ModelTier(
    name="gpt-4o",
    cost_per_prompt_token=0.0000025,    # $2.50/M
    cost_per_completion_token=0.00001,   # $10.00/M
    max_context_window=128_000,
    tier="standard",
)
PREMIUM = ModelTier(
    name="claude-opus",
    cost_per_prompt_token=0.000015,     # $15.00/M
    cost_per_completion_token=0.000075,  # $75.00/M
    max_context_window=200_000,
    tier="premium",
)


def _config_with_tiers(*tiers: ModelTier, **kwargs) -> BudgeteerConfig:
    return BudgeteerConfig(model_tiers=list(tiers), **kwargs)


# ===========================================================================
# StrategyRouter.available
# ===========================================================================


class TestRouterAvailable:
    def test_not_available_without_tiers(self):
        router = StrategyRouter(BudgeteerConfig())
        assert router.available is False

    def test_available_with_one_tier(self):
        router = StrategyRouter(_config_with_tiers(CHEAP))
        assert router.available is True

    def test_available_with_multiple_tiers(self):
        router = StrategyRouter(_config_with_tiers(CHEAP, STANDARD, PREMIUM))
        assert router.available is True


# ===========================================================================
# Candidate generation
# ===========================================================================


class TestCandidateGeneration:
    def test_single_tier_produces_five_candidates(self):
        router = StrategyRouter(_config_with_tiers(CHEAP))
        candidates = router.generate_candidates(StepContext(run_id="r1"))
        assert len(candidates) == 5  # 1 tier × 5 levels

    def test_two_tiers_produce_ten_candidates(self):
        router = StrategyRouter(_config_with_tiers(CHEAP, STANDARD))
        candidates = router.generate_candidates(StepContext(run_id="r1"))
        assert len(candidates) == 10  # 2 tiers × 5 levels

    def test_three_tiers_produce_fifteen_candidates(self):
        router = StrategyRouter(_config_with_tiers(CHEAP, STANDARD, PREMIUM))
        candidates = router.generate_candidates(StepContext(run_id="r1"))
        assert len(candidates) == 15

    def test_quality_decreases_with_degradation_level(self):
        router = StrategyRouter(_config_with_tiers(STANDARD))
        candidates = router.generate_candidates(StepContext(run_id="r1"))
        # With a single tier, quality = 1.0 * level_quality
        for i in range(len(candidates) - 1):
            assert candidates[i].quality_score >= candidates[i + 1].quality_score

    def test_expensive_model_has_higher_quality_than_cheap(self):
        router = StrategyRouter(_config_with_tiers(CHEAP, PREMIUM))
        candidates = router.generate_candidates(StepContext(run_id="r1"))

        # Group by model, compare level-0 quality
        cheap_l0 = [c for c in candidates if c.model == "gpt-4o-mini" and c.degrade_level == 0]
        prem_l0 = [c for c in candidates if c.model == "claude-opus" and c.degrade_level == 0]
        assert len(cheap_l0) == 1 and len(prem_l0) == 1
        assert prem_l0[0].quality_score > cheap_l0[0].quality_score

    def test_max_tokens_decrease_with_degradation(self):
        config = _config_with_tiers(STANDARD, default_max_tokens=1000)
        router = StrategyRouter(config)
        candidates = router.generate_candidates(StepContext(run_id="r1"))

        by_level = {c.degrade_level: c.max_tokens for c in candidates if c.model == "gpt-4o"}
        assert by_level[0] == 1000     # 1.0 × 1000
        assert by_level[1] == 750      # 0.75 × 1000
        assert by_level[2] == 500      # 0.5 × 1000
        assert by_level[3] == 500      # 0.5 × 1000
        assert by_level[4] == 250      # 0.25 × 1000

    def test_retrieval_disabled_at_higher_levels(self):
        router = StrategyRouter(_config_with_tiers(STANDARD))
        candidates = router.generate_candidates(StepContext(run_id="r1"))

        by_level = {c.degrade_level: c for c in candidates}
        assert by_level[0].retrieval_enabled is True
        assert by_level[1].retrieval_enabled is True
        assert by_level[2].retrieval_enabled is True
        assert by_level[3].retrieval_enabled is False
        assert by_level[4].retrieval_enabled is False

    def test_tool_calls_decrease_with_degradation(self):
        router = StrategyRouter(_config_with_tiers(STANDARD))
        candidates = router.generate_candidates(StepContext(run_id="r1"))

        by_level = {c.degrade_level: c.tool_calls_allowed for c in candidates}
        assert by_level[0] == 5
        assert by_level[1] == 3
        assert by_level[2] == 2
        assert by_level[3] == 1
        assert by_level[4] == 0

    def test_uses_config_temperature(self):
        config = _config_with_tiers(CHEAP, default_temperature=0.3)
        router = StrategyRouter(config)
        candidates = router.generate_candidates(StepContext(run_id="r1"))
        assert all(c.temperature == 0.3 for c in candidates)

    def test_context_window_from_tier(self):
        router = StrategyRouter(_config_with_tiers(CHEAP, PREMIUM))
        candidates = router.generate_candidates(StepContext(run_id="r1"))
        cheap_cw = {c.context_window for c in candidates if c.model == "gpt-4o-mini"}
        prem_cw = {c.context_window for c in candidates if c.model == "claude-opus"}
        assert cheap_cw == {128_000}
        assert prem_cw == {200_000}


# ===========================================================================
# Forecasting
# ===========================================================================


class TestForecasting:
    def test_cost_from_tier_pricing(self):
        router = StrategyRouter(_config_with_tiers(STANDARD))
        candidate = CandidateStrategy(
            model="gpt-4o",
            max_tokens=100,
            temperature=0.7,
            context_window=128_000,
            tool_calls_allowed=5,
            retrieval_enabled=True,
            retrieval_top_k=3,
            degrade_level=0,
        )
        ctx = StepContext(run_id="r1")  # no messages → 100 prompt tokens
        router.forecast(candidate, ctx)

        expected_cost = (
            100 * STANDARD.cost_per_prompt_token
            + 100 * STANDARD.cost_per_completion_token
        )
        assert candidate.predicted_prompt_tokens == 100
        assert candidate.predicted_completion_tokens == 100
        assert candidate.predicted_cost_usd == pytest.approx(expected_cost)

    def test_prompt_tokens_from_messages(self):
        router = StrategyRouter(_config_with_tiers(CHEAP))
        candidate = CandidateStrategy(
            model="gpt-4o-mini",
            max_tokens=50,
            temperature=0.7,
            context_window=128_000,
            tool_calls_allowed=0,
            retrieval_enabled=False,
            retrieval_top_k=0,
            degrade_level=4,
        )
        # 400 chars → ~100 tokens
        ctx = StepContext(
            run_id="r1",
            messages=[{"role": "user", "content": "x" * 400}],
        )
        router.forecast(candidate, ctx)
        assert candidate.predicted_prompt_tokens == 100

    def test_empty_messages_baseline(self):
        router = StrategyRouter(_config_with_tiers(CHEAP))
        candidate = CandidateStrategy(
            model="gpt-4o-mini",
            max_tokens=50,
            temperature=0.7,
            context_window=128_000,
            tool_calls_allowed=0,
            retrieval_enabled=False,
            retrieval_top_k=0,
            degrade_level=4,
        )
        ctx = StepContext(run_id="r1")
        router.forecast(candidate, ctx)
        assert candidate.predicted_prompt_tokens == 100  # baseline

    def test_latency_estimation(self):
        router = StrategyRouter(_config_with_tiers(CHEAP))
        candidate = CandidateStrategy(
            model="gpt-4o-mini",
            max_tokens=1000,
            temperature=0.7,
            context_window=128_000,
            tool_calls_allowed=0,
            retrieval_enabled=False,
            retrieval_top_k=0,
            degrade_level=0,
        )
        ctx = StepContext(run_id="r1")
        router.forecast(candidate, ctx)
        # 30 + 1000 * 0.015 = 45.0
        assert candidate.predicted_latency_ms == pytest.approx(45.0)

    def test_unknown_model_skips_forecast(self):
        router = StrategyRouter(_config_with_tiers(CHEAP))
        candidate = CandidateStrategy(
            model="unknown-model",
            max_tokens=100,
            temperature=0.7,
            context_window=128_000,
            tool_calls_allowed=0,
            retrieval_enabled=False,
            retrieval_top_k=0,
            degrade_level=0,
        )
        ctx = StepContext(run_id="r1")
        router.forecast(candidate, ctx)
        assert candidate.predicted_cost_usd == 0.0
        assert candidate.predicted_prompt_tokens == 0


# ===========================================================================
# Selection
# ===========================================================================


class TestSelection:
    def _make_candidate(self, model="m", quality=1.0, cost=0.01,
                        tokens=200, latency=50.0, **kwargs):
        c = CandidateStrategy(
            model=model,
            max_tokens=100,
            temperature=0.7,
            context_window=128_000,
            tool_calls_allowed=kwargs.get("tool_calls_allowed", 5),
            retrieval_enabled=True,
            retrieval_top_k=3,
            degrade_level=kwargs.get("degrade_level", 0),
            predicted_prompt_tokens=tokens // 2,
            predicted_completion_tokens=tokens // 2,
            predicted_cost_usd=cost,
            predicted_latency_ms=latency,
            quality_score=quality,
        )
        return c

    def test_selects_highest_quality_unconstrained(self):
        router = StrategyRouter(BudgeteerConfig())
        candidates = [
            self._make_candidate(quality=0.5),
            self._make_candidate(quality=0.9),
            self._make_candidate(quality=0.3),
        ]
        best = router.select(candidates)
        assert best.quality_score == 0.9

    def test_filters_by_usd(self):
        router = StrategyRouter(BudgeteerConfig())
        candidates = [
            self._make_candidate(quality=1.0, cost=0.10),  # too expensive
            self._make_candidate(quality=0.7, cost=0.04),  # fits
            self._make_candidate(quality=0.3, cost=0.01),  # fits
        ]
        best = router.select(candidates, remaining_usd=0.05)
        assert best.quality_score == 0.7

    def test_filters_by_tokens(self):
        router = StrategyRouter(BudgeteerConfig())
        candidates = [
            self._make_candidate(quality=1.0, tokens=1000),  # too many
            self._make_candidate(quality=0.5, tokens=200),   # fits
        ]
        best = router.select(candidates, remaining_tokens=300)
        assert best.quality_score == 0.5

    def test_filters_by_latency(self):
        router = StrategyRouter(BudgeteerConfig())
        candidates = [
            self._make_candidate(quality=1.0, latency=500.0),  # too slow
            self._make_candidate(quality=0.6, latency=100.0),  # fits
        ]
        best = router.select(candidates, remaining_latency_ms=200.0)
        assert best.quality_score == 0.6

    def test_returns_none_when_nothing_fits(self):
        router = StrategyRouter(BudgeteerConfig())
        candidates = [
            self._make_candidate(cost=1.0),
            self._make_candidate(cost=0.5),
        ]
        assert router.select(candidates, remaining_usd=0.001) is None

    def test_combined_constraints(self):
        router = StrategyRouter(BudgeteerConfig())
        candidates = [
            self._make_candidate(quality=1.0, cost=0.10, latency=500.0),
            self._make_candidate(quality=0.8, cost=0.03, latency=500.0),  # latency fail
            self._make_candidate(quality=0.6, cost=0.03, latency=50.0),   # fits both
            self._make_candidate(quality=0.4, cost=0.01, latency=30.0),   # fits both
        ]
        best = router.select(
            candidates, remaining_usd=0.05, remaining_latency_ms=100.0
        )
        assert best.quality_score == 0.6

    def test_empty_candidates_returns_none(self):
        router = StrategyRouter(BudgeteerConfig())
        assert router.select([]) is None


# ===========================================================================
# to_decision conversion
# ===========================================================================


class TestToDecision:
    def test_converts_all_fields(self):
        router = StrategyRouter(BudgeteerConfig())
        candidate = CandidateStrategy(
            model="gpt-4o",
            max_tokens=512,
            temperature=0.5,
            context_window=64_000,
            tool_calls_allowed=3,
            retrieval_enabled=True,
            retrieval_top_k=2,
            degrade_level=1,
            quality_score=0.85,
        )
        decision = router.to_decision(candidate)
        assert decision.model == "gpt-4o"
        assert decision.max_tokens == 512
        assert decision.temperature == 0.5
        assert decision.context_window == 64_000
        assert decision.tool_calls_allowed == 3
        assert decision.retrieval_enabled is True
        assert decision.retrieval_top_k == 2
        assert decision.degrade_level == 1

    def test_no_degrade_reason_at_level_0(self):
        router = StrategyRouter(BudgeteerConfig())
        candidate = CandidateStrategy(
            model="m", max_tokens=100, temperature=0.7,
            context_window=8192, tool_calls_allowed=5,
            retrieval_enabled=True, retrieval_top_k=3,
            degrade_level=0,
        )
        decision = router.to_decision(candidate)
        assert decision.degrade_reason is None

    def test_degrade_reason_at_higher_levels(self):
        router = StrategyRouter(BudgeteerConfig())
        candidate = CandidateStrategy(
            model="m", max_tokens=100, temperature=0.7,
            context_window=8192, tool_calls_allowed=1,
            retrieval_enabled=False, retrieval_top_k=0,
            degrade_level=3,
        )
        decision = router.to_decision(candidate)
        assert "degrade level 3" in decision.degrade_reason


# ===========================================================================
# Degradation ladder constants
# ===========================================================================


class TestDegradeLadder:
    def test_five_levels(self):
        assert len(DEFAULT_DEGRADE_LADDER) == 5

    def test_quality_decreases_monotonically(self):
        qualities = [p.quality for p in DEFAULT_DEGRADE_LADDER]
        for i in range(len(qualities) - 1):
            assert qualities[i] >= qualities[i + 1]

    def test_level_0_is_full_plan(self):
        p = DEFAULT_DEGRADE_LADDER[0]
        assert p.max_tokens_ratio == 1.0
        assert p.tool_calls == 5
        assert p.retrieval is True
        assert p.quality == 1.0

    def test_level_4_is_strict_minimal(self):
        p = DEFAULT_DEGRADE_LADDER[4]
        assert p.max_tokens_ratio == 0.25
        assert p.tool_calls == 0
        assert p.retrieval is False


# ===========================================================================
# Integrated: PolicyEngine with router (routed path)
# ===========================================================================


class TestRoutedPolicyEngine:
    @pytest.fixture()
    def routed_engine(self, tmp_db):
        """PolicyEngine with model tiers → uses routed evaluation."""
        config = BudgeteerConfig(
            model_tiers=[CHEAP, STANDARD, PREMIUM],
            default_max_tokens=1024,
            storage_path=tmp_db,
        )
        store = TelemetryStore(tmp_db)
        eng = PolicyEngine(config, store)
        yield eng
        store.close()

    def test_picks_best_model_with_budget_room(self, routed_engine):
        """With no budget constraints, should pick the highest quality (premium model, level 0)."""
        ctx = StepContext(run_id="r1")
        decision = routed_engine.evaluate(ctx)
        assert decision.model == "claude-opus"
        assert decision.degrade_level == 0
        assert decision.degrade_reason is None

    def test_downgrades_model_under_tight_usd_budget(self, routed_engine):
        """With a tight USD budget, should pick a cheaper model or higher degradation."""
        ctx = StepContext(
            run_id="r1",
            run_budget=RunBudget(hard_usd_cap=0.001),
            current_run_cost_usd=0.0,
        )
        decision = routed_engine.evaluate(ctx)
        # Should not pick the premium model at level 0 (too expensive)
        assert decision.model != "claude-opus" or decision.degrade_level > 0

    def test_raises_when_budget_fully_consumed(self, routed_engine):
        ctx = StepContext(
            run_id="r1",
            run_budget=RunBudget(hard_usd_cap=1.0),
            current_run_cost_usd=1.0,
        )
        with pytest.raises(BudgetExceededError, match="run_usd"):
            routed_engine.evaluate(ctx)

    def test_raises_when_tokens_consumed(self, routed_engine):
        ctx = StepContext(
            run_id="r1",
            run_budget=RunBudget(hard_token_cap=1000),
            current_run_tokens=1000,
        )
        with pytest.raises(BudgetExceededError, match="run_tokens"):
            routed_engine.evaluate(ctx)

    def test_raises_when_latency_consumed(self, routed_engine):
        ctx = StepContext(
            run_id="r1",
            run_budget=RunBudget(hard_latency_cap_ms=5000),
            elapsed_ms=5000,
        )
        with pytest.raises(BudgetExceededError, match="run_latency"):
            routed_engine.evaluate(ctx)

    def test_caps_tool_calls_from_budget(self, routed_engine):
        ctx = StepContext(
            run_id="r1",
            run_budget=RunBudget(max_tool_calls=2),
            current_run_tool_calls=1,
        )
        decision = routed_engine.evaluate(ctx)
        assert decision.tool_calls_allowed <= 1  # only 1 remaining

    def test_tight_token_budget_selects_degraded(self, routed_engine):
        """With a tight token budget, only higher-degradation candidates fit."""
        ctx = StepContext(
            run_id="r1",
            run_budget=RunBudget(hard_token_cap=800),
            current_run_tokens=100,
        )
        decision = routed_engine.evaluate(ctx)
        # remaining=700; level 0 (1024+100=1124) and level 1 (768+100=868) don't fit
        assert decision.degrade_level >= 2
        assert decision.max_tokens <= 700

    def test_no_feasible_strategy_raises(self, tmp_db):
        """When the remaining budget is so tiny no candidate fits."""
        config = BudgeteerConfig(
            model_tiers=[PREMIUM],  # only expensive model
            default_max_tokens=1024,
            storage_path=tmp_db,
        )
        store = TelemetryStore(tmp_db)
        eng = PolicyEngine(config, store)

        ctx = StepContext(
            run_id="r1",
            run_budget=RunBudget(hard_usd_cap=0.0000001),
            current_run_cost_usd=0.0,
        )
        with pytest.raises(BudgetExceededError, match="no_feasible_strategy"):
            eng.evaluate(ctx)
        store.close()

    def test_daily_budget_integration(self, tmp_db):
        """Routed path also checks daily budget limits."""
        config = BudgeteerConfig(
            model_tiers=[CHEAP],
            storage_path=tmp_db,
        )
        store = TelemetryStore(tmp_db)
        store.record_daily_usage("user", "u1", cost_usd=100.0)
        eng = PolicyEngine(config, store)

        account = BudgetAccount(
            scope=BudgetScope.USER,
            scope_id="u1",
            limit_usd_per_day=100.0,
        )
        ctx = StepContext(run_id="r1", budget_account=account)
        with pytest.raises(BudgetExceededError, match="daily_usd"):
            eng.evaluate(ctx)
        store.close()

    def test_daily_runs_cap_raises(self, tmp_db):
        config = BudgeteerConfig(
            model_tiers=[CHEAP],
            storage_path=tmp_db,
        )
        store = TelemetryStore(tmp_db)
        store.record_daily_usage("user", "u1", runs=5)
        eng = PolicyEngine(config, store)

        account = BudgetAccount(
            scope=BudgetScope.USER,
            scope_id="u1",
            limit_runs_per_day=5,
        )
        ctx = StepContext(run_id="r1", budget_account=account)
        with pytest.raises(BudgetExceededError, match="daily_runs"):
            eng.evaluate(ctx)
        store.close()


# ===========================================================================
# Legacy path unchanged
# ===========================================================================


class TestLegacyPathUnchanged:
    """Verify the legacy path (no model tiers) still behaves identically."""

    def test_defaults_without_tiers(self, tmp_db):
        config = BudgeteerConfig(storage_path=tmp_db)
        store = TelemetryStore(tmp_db)
        eng = PolicyEngine(config, store)
        decision = eng.evaluate(StepContext(run_id="r1"))
        assert decision.model == "gpt-4o-mini"
        assert decision.max_tokens == 1024
        assert decision.degrade_level == 0
        store.close()

    def test_degradation_at_80pct(self, tmp_db):
        config = BudgeteerConfig(storage_path=tmp_db)
        store = TelemetryStore(tmp_db)
        eng = PolicyEngine(config, store)
        ctx = StepContext(
            run_id="r1",
            run_budget=RunBudget(hard_usd_cap=10.0),
            current_run_cost_usd=8.0,
        )
        decision = eng.evaluate(ctx)
        assert decision.degrade_level == 1
        assert decision.max_tokens <= 512
        store.close()
