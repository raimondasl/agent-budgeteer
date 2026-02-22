"""Tests for budgeteer.sdk — end-to-end SDK integration with policy engine."""

import pytest

from budgeteer.config import BudgeteerConfig
from budgeteer.exceptions import BudgetExceededError
from budgeteer.models import (
    BudgetAccount,
    BudgetScope,
    RunBudget,
    StepContext,
    StepMetrics,
)
from budgeteer.sdk import Budgeteer


class TestSDKLifecycle:
    def test_start_and_end_run(self, config):
        sdk = Budgeteer(config)
        run = sdk.start_run()
        assert run.run_id in sdk._active_runs
        result = sdk.end_run(run.run_id, success=True)
        assert result.success is True
        assert run.run_id not in sdk._active_runs
        sdk.close()

    def test_before_step_returns_decision(self, config):
        sdk = Budgeteer(config)
        run = sdk.start_run()
        ctx = StepContext(run_id=run.run_id)
        decision = sdk.before_step(ctx)
        assert decision.model == config.default_model
        sdk.end_run(run.run_id)
        sdk.close()

    def test_after_step_updates_run_totals(self, config):
        sdk = Budgeteer(config)
        run = sdk.start_run()
        ctx = StepContext(run_id=run.run_id)
        decision = sdk.before_step(ctx)
        metrics = StepMetrics(
            prompt_tokens=100,
            completion_tokens=50,
            cost_usd=0.01,
            latency_ms=200,
            tool_calls_made=2,
        )
        sdk.after_step(ctx, decision, metrics)

        assert run.total_cost_usd == 0.01
        assert run.total_tokens == 150
        assert run.total_steps == 1
        assert run.total_tool_calls == 2
        sdk.end_run(run.run_id)
        sdk.close()


class TestSDKRunBudgetEnforcement:
    def test_before_step_auto_populates_budget(self, config):
        sdk = Budgeteer(config)
        budget = RunBudget(hard_usd_cap=5.0)
        run = sdk.start_run(run_budget=budget)

        ctx = StepContext(run_id=run.run_id)
        decision = sdk.before_step(ctx)
        # Should auto-populate from tracked state
        assert ctx.run_budget is budget
        assert ctx.current_run_cost_usd == 0.0
        assert decision.degrade_level == 0
        sdk.end_run(run.run_id)
        sdk.close()

    def test_budget_exceeded_after_spending(self, config):
        sdk = Budgeteer(config)
        budget = RunBudget(hard_usd_cap=0.05)
        run = sdk.start_run(run_budget=budget)

        # Step 1: spend most of the budget
        ctx1 = StepContext(run_id=run.run_id)
        d1 = sdk.before_step(ctx1)
        sdk.after_step(ctx1, d1, StepMetrics(cost_usd=0.05))

        # Step 2: budget is now exhausted
        ctx2 = StepContext(run_id=run.run_id)
        with pytest.raises(BudgetExceededError, match="run_usd"):
            sdk.before_step(ctx2)
        sdk.end_run(run.run_id)
        sdk.close()

    def test_token_cap_reduces_max_tokens(self, config):
        sdk = Budgeteer(config)
        budget = RunBudget(hard_token_cap=500)
        run = sdk.start_run(run_budget=budget)

        # Spend some tokens
        ctx1 = StepContext(run_id=run.run_id)
        d1 = sdk.before_step(ctx1)
        sdk.after_step(
            ctx1, d1, StepMetrics(prompt_tokens=200, completion_tokens=100)
        )

        # Next step should have reduced max_tokens
        ctx2 = StepContext(run_id=run.run_id)
        d2 = sdk.before_step(ctx2)
        assert d2.max_tokens == 200  # 500 - 300 used
        sdk.end_run(run.run_id)
        sdk.close()

    def test_tool_call_limit_enforced(self, config):
        sdk = Budgeteer(config)
        budget = RunBudget(max_tool_calls=3)
        run = sdk.start_run(run_budget=budget)

        ctx1 = StepContext(run_id=run.run_id)
        d1 = sdk.before_step(ctx1)
        sdk.after_step(ctx1, d1, StepMetrics(tool_calls_made=3))

        ctx2 = StepContext(run_id=run.run_id)
        d2 = sdk.before_step(ctx2)
        assert d2.tool_calls_allowed == 0
        sdk.end_run(run.run_id)
        sdk.close()

    def test_default_run_budget_from_config(self, tmp_db):
        default_budget = RunBudget(
            run_id="default", hard_usd_cap=1.0, hard_token_cap=10000
        )
        config = BudgeteerConfig(
            storage_path=tmp_db, default_run_budget=default_budget
        )
        sdk = Budgeteer(config)
        run = sdk.start_run()

        ctx = StepContext(run_id=run.run_id)
        sdk.before_step(ctx)
        assert ctx.run_budget is default_budget
        sdk.end_run(run.run_id)
        sdk.close()

    def test_explicit_budget_overrides_default(self, tmp_db):
        default_budget = RunBudget(
            run_id="default", hard_usd_cap=1.0
        )
        config = BudgeteerConfig(
            storage_path=tmp_db, default_run_budget=default_budget
        )
        sdk = Budgeteer(config)
        custom = RunBudget(hard_usd_cap=50.0)
        run = sdk.start_run(run_budget=custom)

        ctx = StepContext(run_id=run.run_id)
        sdk.before_step(ctx)
        assert ctx.run_budget is custom
        sdk.end_run(run.run_id)
        sdk.close()


class TestSDKDailyBudgetEnforcement:
    def test_daily_ledger_updated_on_start_run(self, config):
        sdk = Budgeteer(config)
        account = BudgetAccount(
            scope=BudgetScope.USER,
            scope_id="u1",
            limit_usd_per_day=100.0,
        )
        sdk.start_run(budget_account=account)

        usage = sdk.telemetry.get_daily_usage("user", "u1")
        assert usage["runs"] == 1
        sdk.close()

    def test_daily_ledger_updated_on_after_step(self, config):
        sdk = Budgeteer(config)
        account = BudgetAccount(
            scope=BudgetScope.USER,
            scope_id="u1",
            limit_usd_per_day=100.0,
        )
        run = sdk.start_run(budget_account=account)

        ctx = StepContext(run_id=run.run_id)
        decision = sdk.before_step(ctx)
        metrics = StepMetrics(
            prompt_tokens=500, completion_tokens=200, cost_usd=0.05
        )
        sdk.after_step(ctx, decision, metrics)

        usage = sdk.telemetry.get_daily_usage("user", "u1")
        assert usage["cost_usd"] == pytest.approx(0.05)
        assert usage["tokens"] == 700
        sdk.close()

    def test_daily_cap_exceeded_raises(self, tmp_db):
        config = BudgeteerConfig(storage_path=tmp_db)
        sdk = Budgeteer(config)
        account = BudgetAccount(
            scope=BudgetScope.USER,
            scope_id="u1",
            limit_usd_per_day=1.0,
        )

        # Simulate prior spending
        sdk.telemetry.record_daily_usage("user", "u1", cost_usd=1.0)

        run = sdk.start_run(budget_account=account)
        ctx = StepContext(run_id=run.run_id)

        with pytest.raises(BudgetExceededError, match="daily_usd"):
            sdk.before_step(ctx)
        sdk.close()

    def test_end_run_cleans_up_budget_state(self, config):
        sdk = Budgeteer(config)
        budget = RunBudget(hard_usd_cap=5.0)
        account = BudgetAccount(
            scope=BudgetScope.USER,
            scope_id="u1",
        )
        run = sdk.start_run(run_budget=budget, budget_account=account)
        sdk.end_run(run.run_id)

        assert run.run_id not in sdk._run_budgets
        assert run.run_id not in sdk._run_accounts
        sdk.close()


class TestSDKGracefulDegradation:
    def test_progressive_degradation(self, config):
        """Verify that degradation increases as budget is consumed."""
        sdk = Budgeteer(config)
        budget = RunBudget(hard_usd_cap=1.0)
        run = sdk.start_run(run_budget=budget)

        # Step at 0% — no degradation
        ctx0 = StepContext(run_id=run.run_id)
        d0 = sdk.before_step(ctx0)
        assert d0.degrade_level == 0
        sdk.after_step(ctx0, d0, StepMetrics(cost_usd=0.80))

        # Step at 80% — level 1
        ctx1 = StepContext(run_id=run.run_id)
        d1 = sdk.before_step(ctx1)
        assert d1.degrade_level == 1
        assert d1.degrade_reason is not None
        sdk.after_step(ctx1, d1, StepMetrics(cost_usd=0.10))

        # Step at 90% — level 2
        ctx2 = StepContext(run_id=run.run_id)
        d2 = sdk.before_step(ctx2)
        assert d2.degrade_level == 2
        assert d2.tool_calls_allowed == 0
        assert d2.retrieval_enabled is False
        sdk.after_step(ctx2, d2, StepMetrics(cost_usd=0.10))

        # Step at 100% — raises
        ctx3 = StepContext(run_id=run.run_id)
        with pytest.raises(BudgetExceededError):
            sdk.before_step(ctx3)

        sdk.end_run(run.run_id)
        sdk.close()
