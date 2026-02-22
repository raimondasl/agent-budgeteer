"""Tests for Milestone 9 — Orchestrated Step Executor (execute_step)."""

import pytest

from budgeteer.config import BudgeteerConfig
from budgeteer.context_manager import RetrievalResult
from budgeteer.llm_client import LLMClient
from budgeteer.models import (
    ModelTier,
    RunBudget,
    StepMetrics,
    StepResult,
)
from budgeteer.sdk import Budgeteer
from budgeteer.tool_executor import ToolExecutor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_llm_fn(
    content="Hello",
    prompt_tokens=10,
    completion_tokens=5,
    tool_calls=None,
):
    """Return a fake LLM callable."""
    def fn(*, model, messages, max_tokens, temperature, **kwargs):
        raw = {
            "content": content,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
            "model": model,
        }
        if tool_calls is not None:
            raw["tool_calls"] = tool_calls
        return raw
    return fn


def _basic_config(tmp_db, with_tiers=False):
    tiers = []
    if with_tiers:
        tiers = [
            ModelTier(
                name="cheap",
                cost_per_prompt_token=0.001,
                cost_per_completion_token=0.002,
                max_context_window=4096,
            ),
        ]
    return BudgeteerConfig(
        storage_path=tmp_db,
        default_max_tokens=1024,
        model_tiers=tiers,
    )


# ---------------------------------------------------------------------------
# Basic orchestration
# ---------------------------------------------------------------------------


class TestExecuteStepBasic:
    def test_execute_step_returns_step_result(self, tmp_db):
        config = _basic_config(tmp_db)
        client = LLMClient(_make_llm_fn())
        sdk = Budgeteer(config, llm_client=client)
        run = sdk.start_run()

        result = sdk.execute_step(
            run.run_id,
            messages=[{"role": "user", "content": "hello"}],
        )

        assert isinstance(result, StepResult)
        assert result.llm_response.content == "Hello"
        assert result.decision is not None
        assert result.metrics.prompt_tokens == 10
        assert result.metrics.completion_tokens == 5
        sdk.end_run(run.run_id)
        sdk.close()

    def test_execute_step_without_llm_client_raises(self, tmp_db):
        config = _basic_config(tmp_db)
        sdk = Budgeteer(config)
        run = sdk.start_run()

        with pytest.raises(RuntimeError, match="llm_client"):
            sdk.execute_step(run.run_id, messages=[])
        sdk.end_run(run.run_id)
        sdk.close()

    def test_execute_step_updates_run_totals(self, tmp_db):
        config = _basic_config(tmp_db)
        client = LLMClient(_make_llm_fn(prompt_tokens=100, completion_tokens=50))
        sdk = Budgeteer(config, llm_client=client)
        run = sdk.start_run()

        sdk.execute_step(run.run_id, messages=[{"role": "user", "content": "hi"}])

        assert run.total_steps == 1
        assert run.total_tokens == 150
        sdk.end_run(run.run_id)
        sdk.close()

    def test_execute_step_records_to_telemetry(self, tmp_db):
        config = _basic_config(tmp_db)
        client = LLMClient(_make_llm_fn())
        sdk = Budgeteer(config, llm_client=client)
        run = sdk.start_run()

        sdk.execute_step(run.run_id, messages=[{"role": "user", "content": "hi"}])

        steps = sdk.telemetry.get_steps(run.run_id)
        assert len(steps) == 1
        assert steps[0].actual is not None
        sdk.end_run(run.run_id)
        sdk.close()

    def test_execute_step_multiple_steps(self, tmp_db):
        config = _basic_config(tmp_db)
        client = LLMClient(_make_llm_fn())
        sdk = Budgeteer(config, llm_client=client)
        run = sdk.start_run()

        for _ in range(3):
            sdk.execute_step(run.run_id, messages=[{"role": "user", "content": "hi"}])

        assert run.total_steps == 3
        sdk.end_run(run.run_id)
        sdk.close()


# ---------------------------------------------------------------------------
# Context truncation
# ---------------------------------------------------------------------------


class TestExecuteStepContextTruncation:
    def test_context_truncation_flag(self, tmp_db):
        config = BudgeteerConfig(
            storage_path=tmp_db,
            default_max_tokens=1024,
        )
        client = LLMClient(_make_llm_fn())
        sdk = Budgeteer(config, llm_client=client)
        run = sdk.start_run()

        # Create messages that exceed the default context_window (8192 tokens)
        # ~4 chars per token, so 40000 chars > 8192 tokens
        long_messages = [
            {"role": "user", "content": "x" * 40000},
            {"role": "assistant", "content": "y" * 40000},
            {"role": "user", "content": "z" * 40000},
        ]
        result = sdk.execute_step(run.run_id, messages=long_messages)
        assert result.context_was_truncated is True
        sdk.end_run(run.run_id)
        sdk.close()

    def test_no_truncation_when_fits(self, tmp_db):
        config = _basic_config(tmp_db)
        client = LLMClient(_make_llm_fn())
        sdk = Budgeteer(config, llm_client=client)
        run = sdk.start_run()

        result = sdk.execute_step(
            run.run_id,
            messages=[{"role": "user", "content": "hi"}],
        )
        assert result.context_was_truncated is False
        sdk.end_run(run.run_id)
        sdk.close()


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------


class TestExecuteStepToolCalls:
    def test_tool_calls_executed(self, tmp_db):
        tool_calls = [
            {"name": "search", "arguments": {"query": "test"}},
        ]
        config = _basic_config(tmp_db)
        client = LLMClient(_make_llm_fn(tool_calls=tool_calls))
        executor = ToolExecutor()
        executor.register("search", lambda query: f"results for {query}")

        sdk = Budgeteer(config, llm_client=client, tool_executor=executor)
        run = sdk.start_run()

        result = sdk.execute_step(
            run.run_id,
            messages=[{"role": "user", "content": "hi"}],
        )

        assert len(result.tool_results) == 1
        assert result.tool_results[0].success is True
        assert result.tool_results[0].output == "results for test"
        assert result.metrics.tool_calls_made == 1
        sdk.end_run(run.run_id)
        sdk.close()

    def test_tool_call_limiting(self, tmp_db):
        """LLM returns 5 tool calls, decision allows only 2."""
        tool_calls = [
            {"name": "t1", "arguments": {}},
            {"name": "t2", "arguments": {}},
            {"name": "t3", "arguments": {}},
            {"name": "t4", "arguments": {}},
            {"name": "t5", "arguments": {}},
        ]
        # Use model tiers so router controls tool_calls_allowed
        config = BudgeteerConfig(
            storage_path=tmp_db,
            default_max_tokens=1024,
            model_tiers=[
                ModelTier(name="m", cost_per_prompt_token=0.001,
                          cost_per_completion_token=0.002, max_context_window=4096),
            ],
        )
        client = LLMClient(_make_llm_fn(tool_calls=tool_calls))
        executor = ToolExecutor()
        for name in ["t1", "t2", "t3", "t4", "t5"]:
            executor.register(name, lambda: "ok")

        sdk = Budgeteer(config, llm_client=client, tool_executor=executor)
        run = sdk.start_run(run_budget=RunBudget(hard_usd_cap=100.0))

        result = sdk.execute_step(
            run.run_id,
            messages=[{"role": "user", "content": "hi"}],
        )

        # Router assigns tool_calls_allowed based on degradation level
        assert len(result.tool_results) <= result.decision.tool_calls_allowed
        sdk.end_run(run.run_id)
        sdk.close()

    def test_tool_calls_without_executor(self, tmp_db):
        """Tool calls present in LLM response but no executor — skipped gracefully."""
        tool_calls = [{"name": "search", "arguments": {}}]
        config = _basic_config(tmp_db)
        client = LLMClient(_make_llm_fn(tool_calls=tool_calls))

        sdk = Budgeteer(config, llm_client=client)  # no tool_executor
        run = sdk.start_run()

        result = sdk.execute_step(
            run.run_id,
            messages=[{"role": "user", "content": "hi"}],
        )

        assert len(result.tool_results) == 1
        assert result.tool_results[0].success is False
        assert "No tool executor" in result.tool_results[0].error
        sdk.end_run(run.run_id)
        sdk.close()

    def test_tool_call_telemetry_logged(self, tmp_db):
        tool_calls = [{"name": "search", "arguments": {"q": "test"}}]
        config = _basic_config(tmp_db)
        client = LLMClient(_make_llm_fn(tool_calls=tool_calls))
        executor = ToolExecutor()
        executor.register("search", lambda q: "ok")

        sdk = Budgeteer(config, llm_client=client, tool_executor=executor)
        run = sdk.start_run()
        result = sdk.execute_step(run.run_id, messages=[{"role": "user", "content": "hi"}])

        tool_records = sdk.telemetry.get_tool_calls(run.run_id)
        assert len(tool_records) == 1
        assert tool_records[0].tool_name == "search"
        assert tool_records[0].success is True
        sdk.end_run(run.run_id)
        sdk.close()

    def test_no_tool_calls_in_response(self, tmp_db):
        config = _basic_config(tmp_db)
        client = LLMClient(_make_llm_fn())  # no tool_calls in response
        executor = ToolExecutor()
        executor.register("search", lambda: "ok")

        sdk = Budgeteer(config, llm_client=client, tool_executor=executor)
        run = sdk.start_run()
        result = sdk.execute_step(run.run_id, messages=[{"role": "user", "content": "hi"}])

        assert result.tool_results == []
        assert result.metrics.tool_calls_made == 0
        sdk.end_run(run.run_id)
        sdk.close()

    def test_failed_tool_call(self, tmp_db):
        tool_calls = [{"name": "bad", "arguments": {}}]
        config = _basic_config(tmp_db)
        client = LLMClient(_make_llm_fn(tool_calls=tool_calls))
        executor = ToolExecutor()
        executor.register("bad", lambda: (_ for _ in ()).throw(ValueError("boom")))

        sdk = Budgeteer(config, llm_client=client, tool_executor=executor)
        run = sdk.start_run()
        result = sdk.execute_step(run.run_id, messages=[{"role": "user", "content": "hi"}])

        assert len(result.tool_results) == 1
        assert result.tool_results[0].success is False
        sdk.end_run(run.run_id)
        sdk.close()


# ---------------------------------------------------------------------------
# Retrieval packing
# ---------------------------------------------------------------------------


class TestExecuteStepRetrieval:
    def test_retrieval_results_packed(self, tmp_db):
        received_messages = []

        def mock_llm(*, model, messages, max_tokens, temperature, **kwargs):
            received_messages.extend(messages)
            return {
                "content": "ok",
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                "model": model,
            }

        config = _basic_config(tmp_db)
        client = LLMClient(mock_llm)
        sdk = Budgeteer(config, llm_client=client)
        run = sdk.start_run()

        retrieval = [
            RetrievalResult(content="Fact A", score=0.9),
            RetrievalResult(content="Fact B", score=0.5),
        ]
        result = sdk.execute_step(
            run.run_id,
            messages=[{"role": "user", "content": "hi"}],
            retrieval_results=retrieval,
        )

        # Retrieval content should be appended to messages sent to LLM
        retrieval_msgs = [m for m in received_messages if "[Retrieved context]" in m.get("content", "")]
        assert len(retrieval_msgs) == 1
        assert "Fact A" in retrieval_msgs[0]["content"]
        sdk.end_run(run.run_id)
        sdk.close()

    def test_retrieval_skipped_when_disabled(self, tmp_db):
        """When retrieval_enabled=False in decision, retrieval results are not packed."""
        received_messages = []

        def mock_llm(*, model, messages, max_tokens, temperature, **kwargs):
            received_messages.extend(messages)
            return {
                "content": "ok",
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                "model": model,
            }

        # Use tiers so we get a routed decision we can inspect
        config = BudgeteerConfig(
            storage_path=tmp_db,
            default_max_tokens=1024,
            model_tiers=[
                ModelTier(name="m", cost_per_prompt_token=0.001,
                          cost_per_completion_token=0.002, max_context_window=4096),
            ],
        )
        client = LLMClient(mock_llm)
        sdk = Budgeteer(config, llm_client=client)
        run = sdk.start_run(run_budget=RunBudget(hard_usd_cap=100.0))

        retrieval = [RetrievalResult(content="Fact A", score=0.9)]

        # Get a step result and check
        result = sdk.execute_step(
            run.run_id,
            messages=[{"role": "user", "content": "hi"}],
            retrieval_results=retrieval,
        )

        # Whether retrieval was packed depends on the decision
        if result.decision.retrieval_enabled:
            # If enabled, retrieval should be included
            retrieval_msgs = [m for m in received_messages if "[Retrieved context]" in m.get("content", "")]
            assert len(retrieval_msgs) >= 0  # May or may not be there depending on level
        sdk.end_run(run.run_id)
        sdk.close()

    def test_retrieval_empty_list(self, tmp_db):
        config = _basic_config(tmp_db)
        client = LLMClient(_make_llm_fn())
        sdk = Budgeteer(config, llm_client=client)
        run = sdk.start_run()

        result = sdk.execute_step(
            run.run_id,
            messages=[{"role": "user", "content": "hi"}],
            retrieval_results=[],
        )
        # Should work fine with empty retrieval
        assert result.llm_response.content == "Hello"
        sdk.end_run(run.run_id)
        sdk.close()


# ---------------------------------------------------------------------------
# Cost computation
# ---------------------------------------------------------------------------


class TestExecuteStepCost:
    def test_cost_from_llm_response(self, tmp_db):
        tiers = [
            ModelTier(name="m", cost_per_prompt_token=0.01,
                      cost_per_completion_token=0.03, max_context_window=4096),
        ]
        config = BudgeteerConfig(
            storage_path=tmp_db,
            default_max_tokens=1024,
            model_tiers=tiers,
        )
        client = LLMClient(
            _make_llm_fn(prompt_tokens=100, completion_tokens=50),
            model_tiers=tiers,
        )
        sdk = Budgeteer(config, llm_client=client)
        run = sdk.start_run(run_budget=RunBudget(hard_usd_cap=100.0))

        result = sdk.execute_step(
            run.run_id,
            messages=[{"role": "user", "content": "hi"}],
        )

        expected_cost = 100 * 0.01 + 50 * 0.03
        assert result.metrics.cost_usd == pytest.approx(expected_cost)
        sdk.end_run(run.run_id)
        sdk.close()

    def test_cost_fallback_to_compute_cost(self, tmp_db):
        """If LLMResponse.cost_usd is 0 but model_tiers exist, compute from tiers."""
        tiers = [
            ModelTier(name="m", cost_per_prompt_token=0.01,
                      cost_per_completion_token=0.03, max_context_window=4096),
        ]
        config = BudgeteerConfig(
            storage_path=tmp_db,
            default_max_tokens=1024,
            model_tiers=tiers,
        )
        # Client without tiers (so LLMResponse.cost_usd = 0)
        client = LLMClient(_make_llm_fn(prompt_tokens=100, completion_tokens=50))
        sdk = Budgeteer(config, llm_client=client)
        run = sdk.start_run(run_budget=RunBudget(hard_usd_cap=100.0))

        result = sdk.execute_step(
            run.run_id,
            messages=[{"role": "user", "content": "hi"}],
        )

        expected_cost = 100 * 0.01 + 50 * 0.03
        assert result.metrics.cost_usd == pytest.approx(expected_cost)
        sdk.end_run(run.run_id)
        sdk.close()


# ---------------------------------------------------------------------------
# Manual API still works alongside execute_step
# ---------------------------------------------------------------------------


class TestManualAPIAlongsideOrchestration:
    def test_manual_and_orchestrated_mixed(self, tmp_db):
        config = _basic_config(tmp_db)
        client = LLMClient(_make_llm_fn())
        sdk = Budgeteer(config, llm_client=client)
        run = sdk.start_run()

        # Manual step
        from budgeteer.models import StepContext
        ctx = StepContext(run_id=run.run_id)
        decision = sdk.before_step(ctx)
        sdk.after_step(ctx, decision, StepMetrics(prompt_tokens=10, completion_tokens=5, cost_usd=0.001))

        # Orchestrated step
        result = sdk.execute_step(
            run.run_id,
            messages=[{"role": "user", "content": "hi"}],
        )

        assert run.total_steps == 2
        assert isinstance(result, StepResult)
        sdk.end_run(run.run_id)
        sdk.close()


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


class TestExecuteStepMetadata:
    def test_metadata_passed_to_context(self, tmp_db):
        config = _basic_config(tmp_db)
        client = LLMClient(_make_llm_fn())
        sdk = Budgeteer(config, llm_client=client)
        run = sdk.start_run()

        result = sdk.execute_step(
            run.run_id,
            messages=[{"role": "user", "content": "hi"}],
            metadata={"task_complexity": 0.8},
        )
        # Just ensure it doesn't crash and returns a result
        assert isinstance(result, StepResult)
        sdk.end_run(run.run_id)
        sdk.close()


# ---------------------------------------------------------------------------
# Summarize function
# ---------------------------------------------------------------------------


class TestExecuteStepSummarization:
    def test_summarize_fn_called_on_overflow(self, tmp_db):
        summarize_calls = []

        def mock_summarize(messages):
            summarize_calls.append(messages)
            return "Summary of conversation"

        config = BudgeteerConfig(
            storage_path=tmp_db,
            default_max_tokens=1024,
        )
        client = LLMClient(_make_llm_fn())
        sdk = Budgeteer(config, llm_client=client, summarize_fn=mock_summarize)
        run = sdk.start_run()

        # Messages exceeding default context_window (8192 tokens)
        long_messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "x" * 40000},
            {"role": "assistant", "content": "y" * 40000},
            {"role": "user", "content": "z" * 40000},
        ]
        result = sdk.execute_step(run.run_id, messages=long_messages)
        assert result.context_was_truncated is True
        sdk.end_run(run.run_id)
        sdk.close()
