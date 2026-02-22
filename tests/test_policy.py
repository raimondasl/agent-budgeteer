"""Tests for budgeteer.policy — the policy engine and budget enforcement."""

import pytest

from budgeteer.config import BudgeteerConfig
from budgeteer.exceptions import BudgetExceededError
from budgeteer.models import (
    BudgetAccount,
    BudgetScope,
    RunBudget,
    StepContext,
)
from budgeteer.policy import PolicyEngine
from budgeteer.telemetry import TelemetryStore


@pytest.fixture()
def engine(tmp_db):
    """A PolicyEngine backed by a temp database."""
    config = BudgeteerConfig()
    store = TelemetryStore(tmp_db)
    eng = PolicyEngine(config, store)
    yield eng
    store.close()


@pytest.fixture()
def store(tmp_db):
    """A standalone TelemetryStore for ledger tests."""
    s = TelemetryStore(tmp_db)
    yield s
    s.close()


# ---------------------------------------------------------------------------
# No budget — defaults
# ---------------------------------------------------------------------------


class TestNoBudget:
    def test_returns_defaults(self, engine):
        ctx = StepContext(run_id="r1")
        decision = engine.evaluate(ctx)
        assert decision.model == "gpt-4o-mini"
        assert decision.max_tokens == 1024
        assert decision.degrade_level == 0
        assert decision.degrade_reason is None

    def test_custom_config_defaults(self, tmp_db):
        config = BudgeteerConfig(
            default_model="claude-sonnet", default_max_tokens=2048
        )
        store = TelemetryStore(tmp_db)
        eng = PolicyEngine(config, store)
        decision = eng.evaluate(StepContext(run_id="r1"))
        assert decision.model == "claude-sonnet"
        assert decision.max_tokens == 2048
        store.close()


# ---------------------------------------------------------------------------
# Per-run hard caps
# ---------------------------------------------------------------------------


class TestRunBudgetUSD:
    def test_under_cap_no_degradation(self, engine):
        ctx = StepContext(
            run_id="r1",
            run_budget=RunBudget(hard_usd_cap=10.0),
            current_run_cost_usd=1.0,
        )
        decision = engine.evaluate(ctx)
        assert decision.degrade_level == 0

    def test_at_80pct_degrades_level_1(self, engine):
        ctx = StepContext(
            run_id="r1",
            run_budget=RunBudget(hard_usd_cap=10.0),
            current_run_cost_usd=8.0,
        )
        decision = engine.evaluate(ctx)
        assert decision.degrade_level == 1
        assert decision.max_tokens <= 512  # half of default 1024
        assert decision.tool_calls_allowed <= 2

    def test_at_90pct_degrades_level_2(self, engine):
        ctx = StepContext(
            run_id="r1",
            run_budget=RunBudget(hard_usd_cap=10.0),
            current_run_cost_usd=9.5,
        )
        decision = engine.evaluate(ctx)
        assert decision.degrade_level == 2
        assert decision.max_tokens <= 256
        assert decision.tool_calls_allowed == 0
        assert decision.retrieval_enabled is False

    def test_at_cap_raises(self, engine):
        ctx = StepContext(
            run_id="r1",
            run_budget=RunBudget(hard_usd_cap=5.0),
            current_run_cost_usd=5.0,
        )
        with pytest.raises(BudgetExceededError, match="run_usd") as exc_info:
            engine.evaluate(ctx)
        assert exc_info.value.limit == 5.0
        assert exc_info.value.current == 5.0

    def test_over_cap_raises(self, engine):
        ctx = StepContext(
            run_id="r1",
            run_budget=RunBudget(hard_usd_cap=5.0),
            current_run_cost_usd=5.5,
        )
        with pytest.raises(BudgetExceededError):
            engine.evaluate(ctx)


class TestRunBudgetTokens:
    def test_caps_max_tokens_to_remaining(self, engine):
        ctx = StepContext(
            run_id="r1",
            run_budget=RunBudget(hard_token_cap=5000),
            current_run_tokens=4800,
        )
        decision = engine.evaluate(ctx)
        assert decision.max_tokens == 200  # 5000 - 4800

    def test_at_cap_raises(self, engine):
        ctx = StepContext(
            run_id="r1",
            run_budget=RunBudget(hard_token_cap=5000),
            current_run_tokens=5000,
        )
        with pytest.raises(BudgetExceededError, match="run_tokens"):
            engine.evaluate(ctx)

    def test_under_cap_keeps_default(self, engine):
        ctx = StepContext(
            run_id="r1",
            run_budget=RunBudget(hard_token_cap=100000),
            current_run_tokens=0,
        )
        decision = engine.evaluate(ctx)
        assert decision.max_tokens == 1024


class TestRunBudgetLatency:
    def test_under_cap_ok(self, engine):
        ctx = StepContext(
            run_id="r1",
            run_budget=RunBudget(hard_latency_cap_ms=30000),
            elapsed_ms=5000,
        )
        decision = engine.evaluate(ctx)
        assert decision.degrade_level == 0

    def test_at_cap_raises(self, engine):
        ctx = StepContext(
            run_id="r1",
            run_budget=RunBudget(hard_latency_cap_ms=10000),
            elapsed_ms=10000,
        )
        with pytest.raises(BudgetExceededError, match="run_latency"):
            engine.evaluate(ctx)


class TestRunBudgetToolCalls:
    def test_under_limit_allows_remaining(self, engine):
        ctx = StepContext(
            run_id="r1",
            run_budget=RunBudget(max_tool_calls=10),
            current_run_tool_calls=7,
        )
        decision = engine.evaluate(ctx)
        assert decision.tool_calls_allowed == 3

    def test_at_limit_blocks_tools(self, engine):
        ctx = StepContext(
            run_id="r1",
            run_budget=RunBudget(max_tool_calls=5),
            current_run_tool_calls=5,
        )
        decision = engine.evaluate(ctx)
        assert decision.tool_calls_allowed == 0

    def test_no_limit_keeps_default(self, engine):
        ctx = StepContext(
            run_id="r1",
            run_budget=RunBudget(),
            current_run_tool_calls=100,
        )
        decision = engine.evaluate(ctx)
        assert decision.tool_calls_allowed == 5  # StepDecision default


# ---------------------------------------------------------------------------
# Combined run budget constraints
# ---------------------------------------------------------------------------


class TestCombinedRunBudget:
    def test_token_and_usd_both_trigger(self, engine):
        ctx = StepContext(
            run_id="r1",
            run_budget=RunBudget(hard_usd_cap=10.0, hard_token_cap=50000),
            current_run_cost_usd=9.0,  # 90% -> level 2
            current_run_tokens=100,
        )
        decision = engine.evaluate(ctx)
        # USD fraction (90%) dominates
        assert decision.degrade_level == 2
        assert decision.retrieval_enabled is False

    def test_highest_degrade_wins(self, engine):
        ctx = StepContext(
            run_id="r1",
            run_budget=RunBudget(
                hard_usd_cap=10.0,
                hard_token_cap=10000,
                hard_latency_cap_ms=60000,
            ),
            current_run_cost_usd=1.0,  # 10% -> 0
            current_run_tokens=8500,   # 85% -> 1
            elapsed_ms=55000,          # 91.7% -> 2
        )
        decision = engine.evaluate(ctx)
        assert decision.degrade_level == 2


# ---------------------------------------------------------------------------
# Per-scope daily limits
# ---------------------------------------------------------------------------


class TestDailyBudget:
    def test_fresh_scope_no_degradation(self, engine):
        account = BudgetAccount(
            scope=BudgetScope.USER,
            scope_id="user-1",
            limit_usd_per_day=100.0,
        )
        ctx = StepContext(run_id="r1", budget_account=account)
        decision = engine.evaluate(ctx)
        assert decision.degrade_level == 0

    def test_daily_usd_at_cap_raises(self, tmp_db):
        config = BudgeteerConfig()
        store = TelemetryStore(tmp_db)
        store.record_daily_usage("user", "user-1", cost_usd=50.0)
        eng = PolicyEngine(config, store)

        account = BudgetAccount(
            scope=BudgetScope.USER,
            scope_id="user-1",
            limit_usd_per_day=50.0,
        )
        ctx = StepContext(run_id="r1", budget_account=account)
        with pytest.raises(BudgetExceededError, match="daily_usd") as exc_info:
            eng.evaluate(ctx)
        assert exc_info.value.scope_id == "user-1"
        store.close()

    def test_daily_tokens_at_cap_raises(self, tmp_db):
        config = BudgeteerConfig()
        store = TelemetryStore(tmp_db)
        store.record_daily_usage("org", "org-1", tokens=1_000_000)
        eng = PolicyEngine(config, store)

        account = BudgetAccount(
            scope=BudgetScope.ORG,
            scope_id="org-1",
            limit_tokens_per_day=1_000_000,
        )
        ctx = StepContext(run_id="r1", budget_account=account)
        with pytest.raises(BudgetExceededError, match="daily_tokens"):
            eng.evaluate(ctx)
        store.close()

    def test_daily_runs_at_cap_raises(self, tmp_db):
        config = BudgeteerConfig()
        store = TelemetryStore(tmp_db)
        store.record_daily_usage("user", "u1", runs=10)
        eng = PolicyEngine(config, store)

        account = BudgetAccount(
            scope=BudgetScope.USER,
            scope_id="u1",
            limit_runs_per_day=10,
        )
        ctx = StepContext(run_id="r1", budget_account=account)
        with pytest.raises(BudgetExceededError, match="daily_runs"):
            eng.evaluate(ctx)
        store.close()

    def test_daily_usd_degrades_at_80pct(self, tmp_db):
        config = BudgeteerConfig()
        store = TelemetryStore(tmp_db)
        store.record_daily_usage("user", "u1", cost_usd=85.0)
        eng = PolicyEngine(config, store)

        account = BudgetAccount(
            scope=BudgetScope.USER,
            scope_id="u1",
            limit_usd_per_day=100.0,
        )
        ctx = StepContext(run_id="r1", budget_account=account)
        decision = eng.evaluate(ctx)
        assert decision.degrade_level == 1
        assert "daily cost" in decision.degrade_reason
        store.close()


# ---------------------------------------------------------------------------
# Degradation level helper
# ---------------------------------------------------------------------------


class TestDegradationLevel:
    def test_below_80_returns_0(self):
        assert PolicyEngine._degradation_level(0.0) == 0
        assert PolicyEngine._degradation_level(0.5) == 0
        assert PolicyEngine._degradation_level(0.79) == 0

    def test_80_to_90_returns_1(self):
        assert PolicyEngine._degradation_level(0.80) == 1
        assert PolicyEngine._degradation_level(0.85) == 1
        assert PolicyEngine._degradation_level(0.899) == 1

    def test_90_plus_returns_2(self):
        assert PolicyEngine._degradation_level(0.90) == 2
        assert PolicyEngine._degradation_level(0.99) == 2
        assert PolicyEngine._degradation_level(1.0) == 2


# ---------------------------------------------------------------------------
# Exception attributes
# ---------------------------------------------------------------------------


class TestBudgetExceededError:
    def test_attributes(self):
        err = BudgetExceededError("run_usd", 5.0, 5.5, scope_id="u1")
        assert err.budget_type == "run_usd"
        assert err.limit == 5.0
        assert err.current == 5.5
        assert err.scope_id == "u1"
        assert "run_usd" in str(err)
        assert "u1" in str(err)

    def test_without_scope(self):
        err = BudgetExceededError("run_tokens", 1000, 1001)
        assert err.scope_id is None
        assert "scope" not in str(err)
