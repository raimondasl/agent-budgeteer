"""Tool executor wrapper that captures execution metrics.

Wraps tool functions to track duration, success/failure, and token usage.
"""

from __future__ import annotations

import time
from typing import Any, Callable

from budgeteer.models import ToolResult


class ToolExecutor:
    """Registry and executor for agent tools with metric tracking.

    Example::

        executor = ToolExecutor()
        executor.register("search", my_search_function)
        result = executor.execute("search", query="agent budgeting")
        print(result.duration_ms, result.success)
    """

    def __init__(self) -> None:
        self._tools: dict[str, Callable[..., Any]] = {}

    def register(self, name: str, fn: Callable[..., Any]) -> None:
        """Register a tool function by name."""
        self._tools[name] = fn

    def list_tools(self) -> list[str]:
        """Return names of all registered tools."""
        return list(self._tools.keys())

    def execute(self, name: str, **kwargs: Any) -> ToolResult:
        """Execute a registered tool and return a result with metrics.

        Returns a ToolResult with success=False if the tool is not found
        or raises an exception.
        """
        if name not in self._tools:
            return ToolResult(
                tool_name=name,
                success=False,
                error=f"Tool '{name}' is not registered",
            )

        start = time.perf_counter()
        try:
            output = self._tools[name](**kwargs)
            duration_ms = (time.perf_counter() - start) * 1000
            return ToolResult(
                tool_name=name,
                output=output,
                success=True,
                duration_ms=duration_ms,
            )
        except Exception as exc:
            duration_ms = (time.perf_counter() - start) * 1000
            return ToolResult(
                tool_name=name,
                success=False,
                error=str(exc),
                duration_ms=duration_ms,
            )
