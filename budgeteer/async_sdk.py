"""Async extension of the Budgeteer SDK.

Provides ``AsyncBudgeteer`` with ``execute_step_async()`` for use in
async/await code.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable

from budgeteer.config import BudgeteerConfig
from budgeteer.llm_client import LLMClient
from budgeteer.models import StepResult
from budgeteer.sdk import Budgeteer
from budgeteer.tool_executor import ToolExecutor


class AsyncBudgeteer(Budgeteer):
    """Async-capable extension of :class:`Budgeteer`.

    Adds ``execute_step_async()`` which runs the orchestrated step
    in a thread pool so it doesn't block the event loop.  All
    synchronous methods from :class:`Budgeteer` remain available.
    """

    def __init__(
        self,
        config: BudgeteerConfig | None = None,
        llm_client: LLMClient | None = None,
        tool_executor: ToolExecutor | None = None,
        summarize_fn: Callable[[list[dict[str, Any]]], str] | None = None,
    ):
        super().__init__(
            config=config,
            llm_client=llm_client,
            tool_executor=tool_executor,
            summarize_fn=summarize_fn,
        )

    async def execute_step_async(
        self,
        run_id: str,
        messages: list[dict[str, Any]],
        retrieval_results: list | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> StepResult:
        """Async version of ``execute_step()``.

        Runs the synchronous ``execute_step()`` in a thread pool executor
        to avoid blocking the event loop.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.execute_step(
                run_id=run_id,
                messages=messages,
                retrieval_results=retrieval_results,
                metadata=metadata,
            ),
        )
