"""Budgeteer SDK — the main entry point.

Provides the core before_step / after_step lifecycle that wraps every agent
step, plus run management and access to telemetry.
"""

from __future__ import annotations

import time
import uuid

from budgeteer.config import BudgeteerConfig
from budgeteer.models import (
    BudgetAccount,
    RunBudget,
    RunRecord,
    StepContext,
    StepDecision,
    StepMetrics,
    StepRecord,
)
from budgeteer.policy import PolicyEngine
from budgeteer.telemetry import TelemetryStore


class Budgeteer:
    """Budget-aware controller for agent steps.

    Usage::

        budgeteer = Budgeteer(config)

        run = budgeteer.start_run(
            run_budget=RunBudget(hard_usd_cap=1.0, hard_token_cap=50000),
        )
        context = StepContext(
            run_id=run.run_id,
            run_budget=run.run_budget,
            current_run_cost_usd=run.total_cost_usd,
            current_run_tokens=run.total_tokens,
        )
        decision = budgeteer.before_step(context)

        # ... execute LLM call using decision.model, decision.max_tokens ...

        metrics = StepMetrics(prompt_tokens=120, completion_tokens=50, cost_usd=0.002)
        budgeteer.after_step(context, decision, metrics)

        budgeteer.end_run(run.run_id, success=True)
    """

    def __init__(self, config: BudgeteerConfig | None = None):
        self._config = config or BudgeteerConfig()
        self._telemetry = TelemetryStore(self._config.storage_path)
        self._policy = PolicyEngine(self._config, self._telemetry)
        self._active_runs: dict[str, RunRecord] = {}
        self._run_budgets: dict[str, RunBudget] = {}
        self._run_accounts: dict[str, BudgetAccount] = {}

    def start_run(
        self,
        run_id: str | None = None,
        run_budget: RunBudget | None = None,
        budget_account: BudgetAccount | None = None,
    ) -> RunRecord:
        """Start tracking a new agent run.

        Args:
            run_id: Optional custom run identifier.
            run_budget: Per-run hard caps (USD, tokens, latency, tool calls).
            budget_account: Per-scope daily limits to enforce.

        Returns a RunRecord that can be used to reference this run.
        """
        run_id = run_id or str(uuid.uuid4())
        # Apply default run budget from config if none provided
        if run_budget is None and self._config.default_run_budget is not None:
            run_budget = self._config.default_run_budget

        record = RunRecord(run_id=run_id)
        self._active_runs[run_id] = record

        if run_budget is not None:
            self._run_budgets[run_id] = run_budget
        if budget_account is not None:
            self._run_accounts[run_id] = budget_account
            self._telemetry.record_daily_usage(
                budget_account.scope.value, budget_account.scope_id, runs=1
            )

        self._telemetry.log_run(record)
        return record

    def before_step(self, context: StepContext) -> StepDecision:
        """Determine the control actions for the next agent step.

        Populates budget context from tracked run state and delegates
        to the policy engine for enforcement and degradation.
        """
        # Auto-populate budget context from tracked state
        run = self._active_runs.get(context.run_id)
        if run is not None:
            context.current_run_cost_usd = run.total_cost_usd
            context.current_run_tokens = run.total_tokens
            context.current_run_tool_calls = run.total_tool_calls
            context.elapsed_ms = (time.time() - run.start_time) * 1000

        if context.run_budget is None and context.run_id in self._run_budgets:
            context.run_budget = self._run_budgets[context.run_id]
        if context.budget_account is None and context.run_id in self._run_accounts:
            context.budget_account = self._run_accounts[context.run_id]

        return self._policy.evaluate(context)

    def after_step(
        self,
        context: StepContext,
        decision: StepDecision,
        metrics: StepMetrics,
    ) -> None:
        """Record observed metrics after a step executes.

        Updates the in-memory run totals, persists a step record
        to the telemetry store, and updates daily budget ledger.
        """
        record = StepRecord(
            run_id=context.run_id,
            step_id=context.step_id,
            decision=decision,
            actual=metrics,
        )
        self._telemetry.log_step(record)

        if context.run_id in self._active_runs:
            run = self._active_runs[context.run_id]
            run.total_cost_usd += metrics.cost_usd
            run.total_tokens += metrics.prompt_tokens + metrics.completion_tokens
            run.total_latency_ms += metrics.latency_ms
            run.total_steps += 1
            run.total_tool_calls += metrics.tool_calls_made

        # Update daily budget ledger
        account = (
            context.budget_account
            or self._run_accounts.get(context.run_id)
        )
        if account is not None:
            self._telemetry.record_daily_usage(
                account.scope.value,
                account.scope_id,
                cost_usd=metrics.cost_usd,
                tokens=metrics.prompt_tokens + metrics.completion_tokens,
            )

    def end_run(self, run_id: str, success: bool | None = None) -> RunRecord:
        """Finalize a run and persist its final state.

        Raises ValueError if the run_id is not active.
        """
        record = self._active_runs.pop(run_id, None)
        if record is None:
            raise ValueError(f"No active run with id '{run_id}'")
        record.end_time = time.time()
        record.success = success
        self._telemetry.update_run(record)
        self._run_budgets.pop(run_id, None)
        self._run_accounts.pop(run_id, None)
        return record

    @property
    def telemetry(self) -> TelemetryStore:
        """Access the underlying telemetry store."""
        return self._telemetry

    @property
    def config(self) -> BudgeteerConfig:
        """Access the current configuration."""
        return self._config

    @property
    def policy(self) -> PolicyEngine:
        """Access the policy engine."""
        return self._policy

    def close(self) -> None:
        """Close the telemetry store connection."""
        self._telemetry.close()
