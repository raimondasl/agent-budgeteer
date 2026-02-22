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
