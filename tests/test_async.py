"""Tests for Milestone 11A — Async support."""

import pytest
import pytest_asyncio

from budgeteer.async_sdk import AsyncBudgeteer
from budgeteer.config import BudgeteerConfig
from budgeteer.llm_client import LLMClient
from budgeteer.models import StepResult
from budgeteer.tool_executor import ToolExecutor


def _make_llm_fn(content="Hello", prompt_tokens=10, completion_tokens=5):
    def fn(*, model, messages, max_tokens, temperature, **kwargs):
        return {
            "content": content,
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
            "model": model,
        }
    return fn


class TestAsyncBudgeteer:
    @pytest.mark.asyncio
    async def test_execute_step_async_basic(self, tmp_db):
        config = BudgeteerConfig(storage_path=tmp_db)
        client = LLMClient(_make_llm_fn())
        sdk = AsyncBudgeteer(config, llm_client=client)
        run = sdk.start_run()

        result = await sdk.execute_step_async(
            run.run_id,
            messages=[{"role": "user", "content": "hello"}],
        )

        assert isinstance(result, StepResult)
        assert result.llm_response.content == "Hello"
        assert result.metrics.prompt_tokens == 10
        sdk.end_run(run.run_id)
        sdk.close()

    @pytest.mark.asyncio
    async def test_execute_step_async_with_tools(self, tmp_db):
        tool_calls = [{"name": "search", "arguments": {"q": "test"}}]
        config = BudgeteerConfig(storage_path=tmp_db)

        def llm_fn(*, model, messages, max_tokens, temperature, **kwargs):
            return {
                "content": "ok",
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                "model": model,
                "tool_calls": tool_calls,
            }

        client = LLMClient(llm_fn)
        executor = ToolExecutor()
        executor.register("search", lambda q: f"results for {q}")

        sdk = AsyncBudgeteer(config, llm_client=client, tool_executor=executor)
        run = sdk.start_run()
        result = await sdk.execute_step_async(
            run.run_id,
            messages=[{"role": "user", "content": "hi"}],
        )

        assert len(result.tool_results) == 1
        assert result.tool_results[0].success is True
        sdk.end_run(run.run_id)
        sdk.close()

    @pytest.mark.asyncio
    async def test_execute_step_async_updates_run(self, tmp_db):
        config = BudgeteerConfig(storage_path=tmp_db)
        client = LLMClient(_make_llm_fn(prompt_tokens=100, completion_tokens=50))
        sdk = AsyncBudgeteer(config, llm_client=client)
        run = sdk.start_run()

        await sdk.execute_step_async(
            run.run_id,
            messages=[{"role": "user", "content": "hi"}],
        )

        assert run.total_steps == 1
        assert run.total_tokens == 150
        sdk.end_run(run.run_id)
        sdk.close()

    @pytest.mark.asyncio
    async def test_async_without_llm_client_raises(self, tmp_db):
        config = BudgeteerConfig(storage_path=tmp_db)
        sdk = AsyncBudgeteer(config)
        run = sdk.start_run()

        with pytest.raises(RuntimeError, match="llm_client"):
            await sdk.execute_step_async(run.run_id, messages=[])
        sdk.end_run(run.run_id)
        sdk.close()

    def test_async_budgeteer_is_budgeteer(self, tmp_db):
        """AsyncBudgeteer inherits all sync methods."""
        from budgeteer.sdk import Budgeteer
        config = BudgeteerConfig(storage_path=tmp_db)
        sdk = AsyncBudgeteer(config)
        assert isinstance(sdk, Budgeteer)
        sdk.close()

    def test_sync_methods_still_work(self, tmp_db):
        config = BudgeteerConfig(storage_path=tmp_db)
        client = LLMClient(_make_llm_fn())
        sdk = AsyncBudgeteer(config, llm_client=client)
        run = sdk.start_run()

        # Sync execute_step should work too
        result = sdk.execute_step(
            run.run_id,
            messages=[{"role": "user", "content": "hi"}],
        )
        assert isinstance(result, StepResult)
        sdk.end_run(run.run_id)
        sdk.close()

    @pytest.mark.asyncio
    async def test_multiple_async_steps(self, tmp_db):
        config = BudgeteerConfig(storage_path=tmp_db)
        client = LLMClient(_make_llm_fn())
        sdk = AsyncBudgeteer(config, llm_client=client)
        run = sdk.start_run()

        for _ in range(3):
            result = await sdk.execute_step_async(
                run.run_id,
                messages=[{"role": "user", "content": "hi"}],
            )

        assert run.total_steps == 3
        sdk.end_run(run.run_id)
        sdk.close()
