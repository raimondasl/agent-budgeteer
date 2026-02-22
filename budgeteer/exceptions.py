"""Custom exceptions for the Budgeteer SDK."""

from __future__ import annotations


class BudgetExceededError(Exception):
    """Raised when a budget hard cap has been reached.

    Attributes:
        budget_type: Which budget was exceeded (e.g. "run_usd", "run_tokens",
            "run_latency", "run_tool_calls", "daily_usd", "daily_tokens",
            "daily_runs").
        limit: The configured cap value.
        current: The current usage value that triggered the error.
        scope_id: Optional identifier for the budget scope (user/org/project).
    """

    def __init__(
        self,
        budget_type: str,
        limit: float | int,
        current: float | int,
        scope_id: str | None = None,
    ):
        self.budget_type = budget_type
        self.limit = limit
        self.current = current
        self.scope_id = scope_id

        parts = [f"Budget exceeded: {budget_type}"]
        if scope_id:
            parts.append(f"(scope={scope_id})")
        parts.append(f"— limit={limit}, current={current}")
        super().__init__(" ".join(parts))


class RetryExhaustedError(Exception):
    """Raised when all retries and fallback attempts are exhausted.

    Attributes:
        attempts: Number of attempts made.
        last_error: The last exception that caused the failure.
        models_tried: List of model names attempted.
    """

    def __init__(
        self,
        attempts: int,
        last_error: Exception,
        models_tried: list[str] | None = None,
    ):
        self.attempts = attempts
        self.last_error = last_error
        self.models_tried = models_tried or []
        models_str = ", ".join(self.models_tried) if self.models_tried else "unknown"
        super().__init__(
            f"All {attempts} attempts exhausted (models tried: {models_str}): {last_error}"
        )
