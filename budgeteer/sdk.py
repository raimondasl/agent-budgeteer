"""Budgeteer SDK — the main entry point.

Provides the core before_step / after_step lifecycle that wraps every agent
step, plus run management and access to telemetry.
"""

from __future__ import annotations

import time
import uuid

from budgeteer.config import BudgeteerConfig
from budgeteer.models import (
    RunRecord,
    StepContext,
    StepDecision,
    StepMetrics,
    StepRecord,
)
from budgeteer.telemetry import TelemetryStore


class Budgeteer:
    """Budget-aware controller for agent steps.

    Usage::

        budgeteer = Budgeteer(config)

        run = budgeteer.start_run()
        context = StepContext(run_id=run.run_id)
        decision = budgeteer.before_step(context)

        # ... execute LLM call using decision.model, decision.max_tokens ...

        metrics = StepMetrics(prompt_tokens=120, completion_tokens=50, cost_usd=0.002)
        budgeteer.after_step(context, decision, metrics)

        budgeteer.end_run(run.run_id, success=True)
    """

    def __init__(self, config: BudgeteerConfig | None = None):
        self._config = config or BudgeteerConfig()
        self._telemetry = TelemetryStore(self._config.storage_path)
        self._active_runs: dict[str, RunRecord] = {}

    def start_run(
        self,
        run_id: str | None = None,
    ) -> RunRecord:
        """Start tracking a new agent run.

        Returns a RunRecord that can be used to reference this run.
        """
        run_id = run_id or str(uuid.uuid4())
        record = RunRecord(run_id=run_id)
        self._active_runs[run_id] = record
        self._telemetry.log_run(record)
        return record

    def before_step(self, context: StepContext) -> StepDecision:
        """Determine the control actions for the next agent step.

        Milestone 1: returns default configuration values.
        Later milestones add policy enforcement, strategy routing, and
        degradation logic.
        """
        return StepDecision(
            model=self._config.default_model,
            max_tokens=self._config.default_max_tokens,
            temperature=self._config.default_temperature,
        )

    def after_step(
        self,
        context: StepContext,
        decision: StepDecision,
        metrics: StepMetrics,
    ) -> None:
        """Record observed metrics after a step executes.

        Updates the in-memory run totals and persists a step record
        to the telemetry store.
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
        return record

    @property
    def telemetry(self) -> TelemetryStore:
        """Access the underlying telemetry store."""
        return self._telemetry

    @property
    def config(self) -> BudgeteerConfig:
        """Access the current configuration."""
        return self._config

    def close(self) -> None:
        """Close the telemetry store connection."""
        self._telemetry.close()
