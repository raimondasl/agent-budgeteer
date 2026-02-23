"""Policy engine for budget enforcement and graceful degradation.

Evaluates budget constraints before each agent step and returns a
StepDecision with appropriate limits applied.  Raises BudgetExceededError
when hard caps have been reached and no further degradation is possible.

When model tiers are configured, delegates to the StrategyRouter for
candidate generation, forecasting, and selection (Milestone 3).
Otherwise falls back to the simple two-level degradation logic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from budgeteer.config import BudgeteerConfig
from budgeteer.exceptions import BudgetExceededError
from budgeteer.models import StepContext, StepDecision, StepMetrics
from budgeteer.router import StrategyRouter
from budgeteer.telemetry import TelemetryStore

if TYPE_CHECKING:
    from budgeteer.calibrator import Calibrator

# Default degradation thresholds (fraction of budget consumed) — legacy path
_DEFAULT_DEGRADE_LEVEL_1 = 0.8  # reduce max_tokens, fewer tool calls
_DEFAULT_DEGRADE_LEVEL_2 = 0.9  # minimal tokens, no tools


class PolicyEngine:
    """Evaluates budget constraints and produces step decisions.

    The engine checks per-run hard caps and per-scope daily limits,
    applying graceful degradation as budgets get tight and raising
    ``BudgetExceededError`` when hard caps are fully consumed.

    When model tiers are configured, the engine uses the
    :class:`StrategyRouter` to generate and select optimal strategies
    under constraints.  Without model tiers, it falls back to simple
    threshold-based degradation.
    """

    def __init__(self, config: BudgeteerConfig, telemetry: TelemetryStore, calibrator: "Calibrator | None" = None):
        self._config = config
        self._telemetry = telemetry
        self._router = StrategyRouter(config, calibrator=calibrator)
        self._last_prediction: StepMetrics | None = None
        # Read configurable thresholds (legacy path)
        self._degrade_level_1 = config.degrade_thresholds[0]
        self._degrade_level_2 = config.degrade_thresholds[1]

    @property
    def last_prediction(self) -> StepMetrics | None:
        """The predicted metrics from the last routed evaluation, or None."""
        return self._last_prediction

    def evaluate(self, context: StepContext) -> StepDecision:
        """Produce a StepDecision given the current run/budget context.

        When model tiers are configured, uses strategy routing to pick
        the best feasible candidate.  Otherwise applies the legacy
        two-level degradation logic.
        """
        if self._router.available:
            return self._evaluate_routed(context)
        return self._evaluate_legacy(context)

    # ------------------------------------------------------------------
    # Routed evaluation (Milestone 3)
    # ------------------------------------------------------------------

    def _evaluate_routed(self, context: StepContext) -> StepDecision:
        """Strategy-based evaluation using the router.

        1. Calculate remaining budget (raises if fully consumed).
        2. Generate and forecast candidates.
        3. Select best feasible candidate.
        4. Apply final caps (tool calls, token remainder).
        """
        remaining = self._calculate_remaining_budget(context)

        candidates = self._router.generate_candidates(context)
        for c in candidates:
            self._router.forecast(c, context)

        selected = self._router.select(
            candidates,
            remaining_usd=remaining.get("usd"),
            remaining_tokens=remaining.get("tokens"),
            remaining_latency_ms=remaining.get("latency_ms"),
        )

        if selected is None:
            self._last_prediction = None
            raise BudgetExceededError("no_feasible_strategy", 0, 0)

        self._last_prediction = self._router.get_prediction(selected)
        decision = self._router.to_decision(selected)

        # Cap tool calls from run budget
        if "tool_calls" in remaining:
            decision.tool_calls_allowed = min(
                decision.tool_calls_allowed, remaining["tool_calls"]
            )

        # Cap max_tokens to remaining token budget
        if "tokens" in remaining:
            decision.max_tokens = min(
                decision.max_tokens, max(1, remaining["tokens"])
            )

        return decision

    def _calculate_remaining_budget(
        self, context: StepContext
    ) -> dict[str, float | int]:
        """Compute remaining budget from run caps and daily limits.

        Raises :class:`BudgetExceededError` immediately when a hard cap
        is fully consumed.  Returns a dict with keys ``usd``, ``tokens``,
        ``latency_ms``, and/or ``tool_calls`` as applicable.
        """
        remaining: dict[str, float | int] = {}

        if context.run_budget:
            budget = context.run_budget

            if budget.hard_usd_cap is not None:
                r = budget.hard_usd_cap - context.current_run_cost_usd
                if r <= 0:
                    raise BudgetExceededError(
                        "run_usd",
                        budget.hard_usd_cap,
                        context.current_run_cost_usd,
                    )
                remaining["usd"] = r

            if budget.hard_token_cap is not None:
                r = budget.hard_token_cap - context.current_run_tokens
                if r <= 0:
                    raise BudgetExceededError(
                        "run_tokens",
                        budget.hard_token_cap,
                        context.current_run_tokens,
                    )
                remaining["tokens"] = r

            if budget.hard_latency_cap_ms is not None:
                r = budget.hard_latency_cap_ms - context.elapsed_ms
                if r <= 0:
                    raise BudgetExceededError(
                        "run_latency",
                        budget.hard_latency_cap_ms,
                        context.elapsed_ms,
                    )
                remaining["latency_ms"] = r

            if budget.max_tool_calls is not None:
                remaining["tool_calls"] = max(
                    0, budget.max_tool_calls - context.current_run_tool_calls
                )

        if context.budget_account:
            account = context.budget_account
            usage = self._telemetry.get_daily_usage(
                account.scope.value, account.scope_id
            )

            if account.limit_usd_per_day is not None:
                current = usage["cost_usd"]
                if current >= account.limit_usd_per_day:
                    raise BudgetExceededError(
                        "daily_usd",
                        account.limit_usd_per_day,
                        current,
                        scope_id=account.scope_id,
                    )
                daily_r = account.limit_usd_per_day - current
                remaining["usd"] = min(
                    remaining.get("usd", float("inf")), daily_r
                )

            if account.limit_tokens_per_day is not None:
                current = usage["tokens"]
                if current >= account.limit_tokens_per_day:
                    raise BudgetExceededError(
                        "daily_tokens",
                        account.limit_tokens_per_day,
                        current,
                        scope_id=account.scope_id,
                    )
                daily_r = account.limit_tokens_per_day - current
                remaining["tokens"] = min(
                    remaining.get("tokens", float("inf")), daily_r
                )

            if account.limit_runs_per_day is not None:
                current = usage["runs"]
                if current >= account.limit_runs_per_day:
                    raise BudgetExceededError(
                        "daily_runs",
                        account.limit_runs_per_day,
                        current,
                        scope_id=account.scope_id,
                    )

        # Clean up infinity sentinels from daily-only constraints
        for k in list(remaining):
            v = remaining[k]
            if isinstance(v, float) and v == float("inf"):
                del remaining[k]

        return remaining

    # ------------------------------------------------------------------
    # Legacy evaluation (Milestones 1–2, no model tiers)
    # ------------------------------------------------------------------

    def _evaluate_legacy(self, context: StepContext) -> StepDecision:
        """Simple two-level degradation without strategy routing."""
        self._last_prediction = None
        decision = StepDecision(
            model=self._config.default_model,
            max_tokens=self._config.default_max_tokens,
            temperature=self._config.default_temperature,
        )

        degrade_level = 0
        degrade_reasons: list[str] = []

        if context.run_budget:
            dl, reasons = self._enforce_run_budget(context, decision)
            if dl > degrade_level:
                degrade_level = dl
            degrade_reasons.extend(reasons)

        if context.budget_account:
            dl, reasons = self._enforce_daily_budget(context, decision)
            if dl > degrade_level:
                degrade_level = dl
            degrade_reasons.extend(reasons)

        decision.degrade_level = degrade_level
        if degrade_reasons:
            decision.degrade_reason = "; ".join(degrade_reasons)

        return decision

    def _enforce_run_budget(
        self, context: StepContext, decision: StepDecision
    ) -> tuple[int, list[str]]:
        """Check per-run caps.  Mutates *decision* in place.

        Returns (degrade_level, list_of_reasons).
        """
        budget = context.run_budget
        degrade_level = 0
        reasons: list[str] = []

        # --- USD cap ---
        if budget.hard_usd_cap is not None:
            if context.current_run_cost_usd >= budget.hard_usd_cap:
                raise BudgetExceededError(
                    "run_usd",
                    budget.hard_usd_cap,
                    context.current_run_cost_usd,
                )
            fraction = context.current_run_cost_usd / budget.hard_usd_cap
            dl = self._degradation_level(fraction)
            if dl > degrade_level:
                degrade_level = dl
                reasons.append(f"run cost at {fraction:.0%} of cap")

        # --- Token cap ---
        if budget.hard_token_cap is not None:
            if context.current_run_tokens >= budget.hard_token_cap:
                raise BudgetExceededError(
                    "run_tokens",
                    budget.hard_token_cap,
                    context.current_run_tokens,
                )
            fraction = context.current_run_tokens / budget.hard_token_cap
            remaining = budget.hard_token_cap - context.current_run_tokens
            if remaining < decision.max_tokens:
                decision.max_tokens = max(1, remaining)
            dl = self._degradation_level(fraction)
            if dl > degrade_level:
                degrade_level = dl
                reasons.append(f"run tokens at {fraction:.0%} of cap")

        # --- Latency cap ---
        if budget.hard_latency_cap_ms is not None:
            if context.elapsed_ms >= budget.hard_latency_cap_ms:
                raise BudgetExceededError(
                    "run_latency",
                    budget.hard_latency_cap_ms,
                    context.elapsed_ms,
                )
            fraction = context.elapsed_ms / budget.hard_latency_cap_ms
            dl = self._degradation_level(fraction)
            if dl > degrade_level:
                degrade_level = dl
                reasons.append(f"run latency at {fraction:.0%} of cap")

        # --- Tool calls cap ---
        if budget.max_tool_calls is not None:
            if context.current_run_tool_calls >= budget.max_tool_calls:
                decision.tool_calls_allowed = 0
                reasons.append("tool call limit reached")
            else:
                remaining = budget.max_tool_calls - context.current_run_tool_calls
                if remaining < decision.tool_calls_allowed:
                    decision.tool_calls_allowed = remaining

        # Apply degradation adjustments
        if degrade_level >= 2:
            decision.max_tokens = min(decision.max_tokens, 256)
            decision.tool_calls_allowed = 0
            decision.retrieval_enabled = False
        elif degrade_level >= 1:
            decision.max_tokens = min(
                decision.max_tokens, self._config.default_max_tokens // 2
            )
            decision.tool_calls_allowed = min(decision.tool_calls_allowed, 2)
            decision.retrieval_top_k = 1

        return degrade_level, reasons

    def _enforce_daily_budget(
        self, context: StepContext, decision: StepDecision
    ) -> tuple[int, list[str]]:
        """Check per-scope daily limits.  Mutates *decision* in place.

        Returns (degrade_level, list_of_reasons).
        """
        account = context.budget_account
        usage = self._telemetry.get_daily_usage(
            account.scope.value, account.scope_id
        )

        degrade_level = 0
        reasons: list[str] = []

        # --- Daily USD ---
        if account.limit_usd_per_day is not None:
            current = usage["cost_usd"]
            if current >= account.limit_usd_per_day:
                raise BudgetExceededError(
                    "daily_usd",
                    account.limit_usd_per_day,
                    current,
                    scope_id=account.scope_id,
                )
            fraction = current / account.limit_usd_per_day
            dl = self._degradation_level(fraction)
            if dl > degrade_level:
                degrade_level = dl
                reasons.append(
                    f"daily cost at {fraction:.0%} of limit "
                    f"(scope={account.scope_id})"
                )

        # --- Daily tokens ---
        if account.limit_tokens_per_day is not None:
            current = usage["tokens"]
            if current >= account.limit_tokens_per_day:
                raise BudgetExceededError(
                    "daily_tokens",
                    account.limit_tokens_per_day,
                    current,
                    scope_id=account.scope_id,
                )
            fraction = current / account.limit_tokens_per_day
            dl = self._degradation_level(fraction)
            if dl > degrade_level:
                degrade_level = dl
                reasons.append(
                    f"daily tokens at {fraction:.0%} of limit "
                    f"(scope={account.scope_id})"
                )

        # --- Daily runs ---
        if account.limit_runs_per_day is not None:
            current = usage["runs"]
            if current >= account.limit_runs_per_day:
                raise BudgetExceededError(
                    "daily_runs",
                    account.limit_runs_per_day,
                    current,
                    scope_id=account.scope_id,
                )

        # Apply degradation adjustments for daily limits
        if degrade_level >= 2:
            decision.max_tokens = min(decision.max_tokens, 256)
            decision.tool_calls_allowed = 0
            decision.retrieval_enabled = False
        elif degrade_level >= 1:
            decision.max_tokens = min(
                decision.max_tokens, self._config.default_max_tokens // 2
            )
            decision.tool_calls_allowed = min(decision.tool_calls_allowed, 2)
            decision.retrieval_top_k = 1

        return degrade_level, reasons

    def _degradation_level(self, fraction: float) -> int:
        """Map budget-usage fraction to a degradation level."""
        if fraction >= self._degrade_level_2:
            return 2
        if fraction >= self._degrade_level_1:
            return 1
        return 0
