"""Budgeteer SDK — the main entry point.

Provides the core before_step / after_step lifecycle that wraps every agent
step, plus run management and access to telemetry.

The optional ``execute_step()`` method automates the full lifecycle:
context management -> LLM call -> tool execution -> metrics -> telemetry.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Callable

from budgeteer.calibrator import Calibrator
from budgeteer.config import BudgeteerConfig
from budgeteer.context_manager import ContextManager
from budgeteer.exceptions import RetryExhaustedError
from budgeteer.llm_client import LLMClient
from budgeteer.models import (
    BudgetAccount,
    LLMResponse,
    RunBudget,
    RunRecord,
    StepContext,
    StepDecision,
    StepMetrics,
    StepRecord,
    StepResult,
    ToolRecord,
    ToolResult,
    compute_cost,
)
from budgeteer.policy import PolicyEngine
from budgeteer.reporting import FullReport, Reporter
from budgeteer.roi import ClarifyingQuestion, ROIEvaluator, ROISignals
from budgeteer.telemetry import TelemetryStore
from budgeteer.tool_executor import ToolExecutor


class Budgeteer:
    """Budget-aware controller for agent steps.

    Usage (manual)::

        budgeteer = Budgeteer(config)
        run = budgeteer.start_run(run_budget=RunBudget(hard_usd_cap=1.0))
        context = StepContext(run_id=run.run_id)
        decision = budgeteer.before_step(context)
        # ... execute LLM call ...
        metrics = StepMetrics(prompt_tokens=120, completion_tokens=50, cost_usd=0.002)
        budgeteer.after_step(context, decision, metrics)
        budgeteer.end_run(run.run_id, success=True)

    Usage (orchestrated)::

        budgeteer = Budgeteer(config, llm_client=client)
        run = budgeteer.start_run()
        result = budgeteer.execute_step(run.run_id, messages=[...])
    """

    def __init__(
        self,
        config: BudgeteerConfig | None = None,
        llm_client: LLMClient | None = None,
        tool_executor: ToolExecutor | None = None,
        summarize_fn: Callable[[list[dict[str, Any]]], str] | None = None,
    ):
        self._config = config or BudgeteerConfig()
        self._telemetry = TelemetryStore(self._config.storage_path)
        self._calibrator: Calibrator | None = None
        if self._config.calibration_enabled:
            self._calibrator = Calibrator(alpha=self._config.calibration_alpha)
        self._policy = PolicyEngine(self._config, self._telemetry, calibrator=self._calibrator)
        self._active_runs: dict[str, RunRecord] = {}
        self._run_budgets: dict[str, RunBudget] = {}
        self._run_accounts: dict[str, BudgetAccount] = {}
        self._pending_predictions: dict[str, StepMetrics] = {}

        # Optional orchestration components
        self._llm_client = llm_client
        self._tool_executor = tool_executor
        self._context_manager = ContextManager(
            max_tokens=self._config.default_max_tokens,
            summarize_fn=summarize_fn,
        )

        # ROI evaluator
        self._roi: ROIEvaluator | None = None
        if self._config.roi_enabled:
            self._roi = ROIEvaluator(
                lambda_latency=self._config.roi_lambda_latency,
                recommend_threshold=self._config.roi_recommend_threshold,
                budget_floor=self._config.roi_budget_floor,
                clarify_ambiguity_threshold=self._config.roi_clarify_ambiguity_threshold,
            )

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

        decision = self._policy.evaluate(context)

        # Store prediction from the routed path for after_step calibration
        prediction = self._policy.last_prediction
        if prediction is not None:
            self._pending_predictions[context.step_id] = prediction

        return decision

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
        predicted = self._pending_predictions.pop(context.step_id, None)
        record = StepRecord(
            run_id=context.run_id,
            step_id=context.step_id,
            decision=decision,
            predicted=predicted,
            actual=metrics,
        )
        self._telemetry.log_step(record)

        # Feed actuals back to calibrator for continuous learning
        if self._calibrator is not None and predicted is not None:
            self._calibrator.update(decision.model, predicted, metrics)

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

    def execute_step(
        self,
        run_id: str,
        messages: list[dict[str, Any]],
        retrieval_results: list | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> StepResult:
        """Execute a full orchestrated step: context -> LLM -> tools -> telemetry.

        Requires ``llm_client`` to be set. Raises ``RuntimeError`` if not.

        Args:
            run_id: Active run identifier.
            messages: Conversation messages for the LLM.
            retrieval_results: Optional retrieval results for packing.
            metadata: Optional metadata dict for the step context.

        Returns a :class:`StepResult` with decision, LLM response, tool
        results, and aggregated metrics.
        """
        if self._llm_client is None:
            raise RuntimeError(
                "execute_step() requires an llm_client. "
                "Pass llm_client to the Budgeteer constructor."
            )

        # 1. Build context and get decision
        ctx = StepContext(
            run_id=run_id,
            messages=list(messages),
            metadata=metadata or {},
        )
        decision = self.before_step(ctx)

        # 2. Truncate messages via ContextManager
        original_count = len(messages)
        truncated_messages = self._context_manager.fit(
            messages, max_tokens=decision.context_window
        )
        context_was_truncated = len(truncated_messages) < original_count

        # 3. ROI signals (computed once, used for retrieval + tool gating)
        roi_decisions: dict[str, Any] = {}
        signals: ROISignals | None = None
        if self._roi is not None:
            signals = self._derive_roi_signals(run_id, ctx, decision)

        # 4. Pack retrieval results if enabled (with ROI gating)
        retrieval_included = False
        if decision.retrieval_enabled and retrieval_results:
            include_retrieval = True
            if self._roi is not None and signals is not None:
                # Estimate retrieval cost as a fraction of step cost
                pred = self._policy.last_prediction
                est_cost = pred.cost_usd * 0.1 if pred else 0.001
                est_latency = pred.latency_ms * 0.1 if pred else 50.0
                roi_result = self._roi.evaluate_retrieval(est_cost, est_latency, signals)
                roi_decisions["retrieval"] = {
                    "roi_score": roi_result.roi_score,
                    "recommended": roi_result.recommended,
                    "reason": roi_result.reason,
                }
                if not roi_result.recommended:
                    include_retrieval = False

            if include_retrieval:
                packed = self._context_manager.pack_retrieval(
                    retrieval_results, top_k=decision.retrieval_top_k
                )
                if packed:
                    retrieval_text = "\n\n".join(r.content for r in packed)
                    truncated_messages.append({
                        "role": "system",
                        "content": f"[Retrieved context]\n{retrieval_text}",
                    })
                    retrieval_included = True

        # 5. Call LLM (with retry/fallback)
        llm_response = self._call_llm_with_retry(
            decision=decision,
            messages=truncated_messages,
        )

        # 6. Execute tool calls (with ROI gating)
        tool_results: list[ToolResult] = []
        raw = llm_response.raw_response
        tool_calls_data = []
        if isinstance(raw, dict):
            tool_calls_data = raw.get("tool_calls", [])

        tools_executed = 0
        for tc in tool_calls_data:
            if tools_executed >= decision.tool_calls_allowed:
                break
            tool_name = tc.get("name", "")
            tool_args = tc.get("arguments", {})

            # ROI gating for tool calls
            if self._roi is not None and signals is not None:
                pred = self._policy.last_prediction
                est_cost = pred.cost_usd * 0.05 if pred else 0.001
                est_latency = pred.latency_ms * 0.05 if pred else 100.0
                tool_roi = self._roi.evaluate_tool_call(est_cost, est_latency, signals)
                roi_decisions[f"tool:{tool_name}"] = {
                    "roi_score": tool_roi.roi_score,
                    "recommended": tool_roi.recommended,
                    "reason": tool_roi.reason,
                }
                if not tool_roi.recommended:
                    continue

            if self._tool_executor is not None:
                result = self._tool_executor.execute(tool_name, **tool_args)
            else:
                result = ToolResult(
                    tool_name=tool_name,
                    success=False,
                    error="No tool executor configured",
                )

            tool_results.append(result)
            tools_executed += 1

            # Log tool call to telemetry
            self._telemetry.log_tool_call(ToolRecord(
                run_id=run_id,
                step_id=ctx.step_id,
                tool_name=result.tool_name,
                duration_ms=result.duration_ms,
                tokens_used=result.tokens_used,
                success=result.success,
                error=result.error,
            ))

        # 6. Build metrics
        total_tool_tokens = sum(r.tokens_used for r in tool_results)
        total_tool_duration = sum(r.duration_ms for r in tool_results)
        cost = llm_response.cost_usd
        if cost == 0.0 and self._config.model_tiers:
            cost = compute_cost(
                llm_response.model,
                llm_response.prompt_tokens,
                llm_response.completion_tokens,
                self._config.model_tiers,
            )

        metrics = StepMetrics(
            prompt_tokens=llm_response.prompt_tokens,
            completion_tokens=llm_response.completion_tokens,
            tool_tokens=total_tool_tokens,
            cost_usd=cost,
            latency_ms=llm_response.latency_ms + total_tool_duration,
            tool_calls_made=len(tool_results),
        )

        # 7. Call after_step
        self.after_step(ctx, decision, metrics)

        return StepResult(
            decision=decision,
            llm_response=llm_response,
            tool_results=tool_results,
            metrics=metrics,
            context_was_truncated=context_was_truncated,
            roi_decisions=roi_decisions if roi_decisions else None,
        )

    def _call_llm_with_retry(
        self,
        decision: StepDecision,
        messages: list[dict[str, Any]],
    ) -> LLMResponse:
        """Call LLM with retry and optional fallback to cheaper models.

        When ``max_retries > 0``, retries the same model on failure.
        When retries are exhausted and ``fallback_enabled`` is True,
        tries progressively cheaper model tiers.
        Raises ``RetryExhaustedError`` if all attempts fail.
        """
        max_retries = self._config.max_retries
        delay_s = self._config.retry_delay_ms / 1000.0

        # Build list of models to try: primary model + fallbacks
        models_to_try = [decision.model]
        if max_retries > 0 and self._config.fallback_enabled and self._config.model_tiers:
            # Sort tiers cheapest first, add those cheaper than current
            sorted_tiers = sorted(
                self._config.model_tiers,
                key=lambda t: t.cost_per_prompt_token + t.cost_per_completion_token,
            )
            for tier in sorted_tiers:
                if tier.name != decision.model and tier.name not in models_to_try:
                    models_to_try.append(tier.name)

        models_tried: list[str] = []
        last_error: Exception | None = None
        total_attempts = 0

        for model in models_to_try:
            for attempt in range(max_retries + 1):
                total_attempts += 1
                models_tried.append(model)
                try:
                    return self._llm_client.complete(
                        model=model,
                        messages=messages,
                        max_tokens=decision.max_tokens,
                        temperature=decision.temperature,
                    )
                except Exception as exc:
                    last_error = exc
                    if attempt < max_retries and delay_s > 0:
                        time.sleep(delay_s)

            # If no retries configured, don't try fallbacks
            if max_retries == 0:
                break

        # If we get here without returning, all attempts failed
        if last_error is not None:
            if max_retries > 0:
                raise RetryExhaustedError(
                    attempts=total_attempts,
                    last_error=last_error,
                    models_tried=models_tried,
                )
            raise last_error

        # Should not reach here, but just in case
        raise RuntimeError("No LLM call attempted")

    def _derive_roi_signals(
        self, run_id: str, context: StepContext, decision: StepDecision
    ) -> ROISignals:
        """Derive ROI signals from the current run state and context metadata."""
        # Remaining budget fraction
        remaining_fraction = 1.0
        budget = context.run_budget or self._run_budgets.get(run_id)
        run = self._active_runs.get(run_id)
        if budget and run:
            if budget.hard_usd_cap and budget.hard_usd_cap > 0:
                remaining_fraction = max(0.0, 1.0 - run.total_cost_usd / budget.hard_usd_cap)
            elif budget.hard_token_cap and budget.hard_token_cap > 0:
                remaining_fraction = max(0.0, 1.0 - run.total_tokens / budget.hard_token_cap)

        # Count previous failures from telemetry
        previous_failures = 0
        tool_records = self._telemetry.get_tool_calls(run_id)
        for tr in tool_records:
            if not tr.success:
                previous_failures += 1

        # Read signals from context metadata
        meta = context.metadata or {}
        ambiguity = float(meta.get("ambiguity", 0.5))
        task_complexity = float(meta.get("task_complexity", 0.5))
        freshness_need = float(meta.get("freshness_need", 0.0))

        return ROISignals(
            ambiguity=ambiguity,
            freshness_need=freshness_need,
            previous_failures=previous_failures,
            remaining_budget_fraction=remaining_fraction,
            task_complexity=task_complexity,
        )

    def suggest_clarification(
        self,
        run_id: str,
        metadata: dict[str, Any] | None = None,
        exclude_categories: set[str] | None = None,
    ) -> ClarifyingQuestion | None:
        """Suggest a clarifying question based on current run state, or None.

        Returns None if ROI is disabled or if clarification is not recommended.
        """
        if self._roi is None:
            return None

        ctx = StepContext(run_id=run_id, metadata=metadata or {})
        decision = StepDecision(model=self._config.default_model)
        signals = self._derive_roi_signals(run_id, ctx, decision)
        return self._roi.select_question(signals, exclude_categories=exclude_categories)

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
    def calibrator(self) -> Calibrator | None:
        """Access the calibrator, or None if calibration is disabled."""
        return self._calibrator

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

    def report(
        self,
        run_ids: list[str] | None = None,
        budget_caps: dict[str, float] | None = None,
    ) -> FullReport:
        """Generate a full report for the given (or all) runs.

        Args:
            run_ids: Run IDs to include. If None, all runs in the store.
            budget_caps: Optional caps dict for compliance checking
                (e.g. ``{"usd": 1.0, "tokens": 50000}``).
        """
        if run_ids is None:
            run_ids = self._telemetry.list_run_ids()
        reporter = Reporter(self._telemetry)
        return reporter.full_report(run_ids, budget_caps=budget_caps)

    def __enter__(self) -> "Budgeteer":
        """Support use as a context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Close resources on context manager exit."""
        self.close()

    def close(self) -> None:
        """Close the telemetry store connection."""
        self._telemetry.close()
