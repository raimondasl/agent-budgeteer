"""Tests for Milestone 11C–D — Streaming, context manager, thread safety, export."""

import json
import threading

import pytest

from budgeteer.config import BudgeteerConfig
from budgeteer.llm_client import LLMClient
from budgeteer.models import (
    LLMResponse,
    ModelTier,
    RunRecord,
    StepContext,
    StepMetrics,
    ToolRecord,
)
from budgeteer.sdk import Budgeteer
from budgeteer.telemetry import TelemetryStore


# ---------------------------------------------------------------------------
# Streaming (LLMClient.complete_stream)
# ---------------------------------------------------------------------------


class TestStreaming:
    def test_complete_stream_yields_chunks(self):
        chunks = [
            {"content": "Hello "},
            {"content": "world", "usage": {"prompt_tokens": 10, "completion_tokens": 5}, "model": "m"},
        ]

        def fn(*, model, messages, max_tokens, temperature, stream=False, **kwargs):
            assert stream is True
            return iter(chunks)

        client = LLMClient(fn)
        gen = client.complete_stream("m", [{"role": "user", "content": "hi"}])

        received = []
        for chunk in gen:
            received.append(chunk)

        assert len(received) == 2
        assert received[0]["content"] == "Hello "
        assert received[1]["content"] == "world"

    def test_complete_stream_tracks_metrics(self):
        chunks = [
            {"content": "Hello "},
            {"content": "world", "usage": {"prompt_tokens": 15, "completion_tokens": 8}, "model": "m1"},
        ]

        def fn(*, model, messages, max_tokens, temperature, stream=False, **kwargs):
            return iter(chunks)

        client = LLMClient(fn)
        gen = client.complete_stream("m1", [])

        # Consume all chunks
        for _ in gen:
            pass

        assert client.call_count == 1
        assert client.total_prompt_tokens == 15
        assert client.total_completion_tokens == 8

    def test_complete_stream_with_cost(self):
        tiers = [
            ModelTier(name="m", cost_per_prompt_token=0.01,
                      cost_per_completion_token=0.03, max_context_window=4096),
        ]
        chunks = [
            {"content": "ok", "usage": {"prompt_tokens": 10, "completion_tokens": 5}, "model": "m"},
        ]

        def fn(*, model, messages, max_tokens, temperature, stream=False, **kwargs):
            return iter(chunks)

        client = LLMClient(fn, model_tiers=tiers)
        gen = client.complete_stream("m", [])
        for _ in gen:
            pass

        expected = 10 * 0.01 + 5 * 0.03
        assert client.total_cost_usd == pytest.approx(expected)

    def test_complete_stream_empty(self):
        def fn(*, model, messages, max_tokens, temperature, stream=False, **kwargs):
            return iter([])

        client = LLMClient(fn)
        gen = client.complete_stream("m", [])
        received = list(gen)
        assert received == []
        # Should still increment call count
        assert client.call_count == 1


# ---------------------------------------------------------------------------
# Context manager (with statement)
# ---------------------------------------------------------------------------


class TestContextManager:
    def test_budgeteer_as_context_manager(self, tmp_db):
        config = BudgeteerConfig(storage_path=tmp_db)
        with Budgeteer(config) as sdk:
            run = sdk.start_run()
            assert run.run_id is not None
            sdk.end_run(run.run_id)
        # After __exit__, close() has been called
        # Accessing telemetry should still have data
        # (SQLite connection is closed but data was committed)

    def test_context_manager_closes_on_exception(self, tmp_db):
        config = BudgeteerConfig(storage_path=tmp_db)
        try:
            with Budgeteer(config) as sdk:
                run = sdk.start_run()
                raise ValueError("test error")
        except ValueError:
            pass
        # Should not raise — close() was called

    def test_context_manager_with_llm_client(self, tmp_db):
        def fn(*, model, messages, max_tokens, temperature, **kwargs):
            return {"content": "ok", "usage": {"prompt_tokens": 1, "completion_tokens": 1}, "model": model}

        config = BudgeteerConfig(storage_path=tmp_db)
        with Budgeteer(config, llm_client=LLMClient(fn)) as sdk:
            run = sdk.start_run()
            result = sdk.execute_step(run.run_id, messages=[{"role": "user", "content": "hi"}])
            assert result.llm_response.content == "ok"
            sdk.end_run(run.run_id)


# ---------------------------------------------------------------------------
# Thread safety (TelemetryStore)
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_writes(self, tmp_db):
        """Multiple threads writing to telemetry concurrently."""
        store = TelemetryStore(tmp_db)
        errors = []

        def writer(thread_id):
            try:
                for i in range(10):
                    run_id = f"run-{thread_id}-{i}"
                    store.log_run(RunRecord(run_id=run_id, start_time=float(i)))
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=writer, args=(t,)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        all_ids = store.list_run_ids()
        assert len(all_ids) == 40  # 4 threads * 10 runs
        store.close()

    def test_concurrent_tool_call_logging(self, tmp_db):
        store = TelemetryStore(tmp_db)
        store.log_run(RunRecord(run_id="r1", start_time=1.0))
        errors = []

        def writer(thread_id):
            try:
                for i in range(10):
                    store.log_tool_call(ToolRecord(
                        run_id="r1",
                        step_id=f"s-{thread_id}-{i}",
                        tool_name=f"tool-{thread_id}",
                        duration_ms=1.0,
                    ))
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=writer, args=(t,)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        tools = store.get_tool_calls("r1")
        assert len(tools) == 40
        store.close()

    def test_has_lock_attribute(self, tmp_db):
        store = TelemetryStore(tmp_db)
        assert hasattr(store, "_lock")
        assert isinstance(store._lock, type(threading.Lock()))
        store.close()


# ---------------------------------------------------------------------------
# export_json
# ---------------------------------------------------------------------------


class TestExportJson:
    def test_export_empty(self, tmp_db):
        store = TelemetryStore(tmp_db)
        result = store.export_json()
        data = json.loads(result)
        assert data == []
        store.close()

    def test_export_single_run(self, tmp_db):
        store = TelemetryStore(tmp_db)
        store.log_run(RunRecord(run_id="r1", start_time=1.0))
        result = store.export_json()
        data = json.loads(result)
        assert len(data) == 1
        assert data[0]["run_id"] == "r1"
        assert "steps" in data[0]
        assert "tool_calls" in data[0]
        store.close()

    def test_export_with_steps_and_tools(self, tmp_db):
        from budgeteer.models import StepDecision, StepRecord
        store = TelemetryStore(tmp_db)
        store.log_run(RunRecord(run_id="r1", start_time=1.0))
        store.log_step(StepRecord(
            run_id="r1", step_id="s1",
            decision=StepDecision(model="m"),
            actual=StepMetrics(prompt_tokens=10, completion_tokens=5, cost_usd=0.01),
        ))
        store.log_tool_call(ToolRecord(
            run_id="r1", step_id="s1",
            tool_name="search", duration_ms=50.0,
        ))

        result = store.export_json()
        data = json.loads(result)
        assert len(data) == 1
        assert len(data[0]["steps"]) == 1
        assert data[0]["steps"][0]["step_id"] == "s1"
        assert len(data[0]["tool_calls"]) == 1
        assert data[0]["tool_calls"][0]["tool_name"] == "search"
        store.close()

    def test_export_specific_run_ids(self, tmp_db):
        store = TelemetryStore(tmp_db)
        store.log_run(RunRecord(run_id="r1", start_time=1.0))
        store.log_run(RunRecord(run_id="r2", start_time=2.0))

        result = store.export_json(run_ids=["r1"])
        data = json.loads(result)
        assert len(data) == 1
        assert data[0]["run_id"] == "r1"
        store.close()

    def test_export_is_valid_json(self, tmp_db):
        store = TelemetryStore(tmp_db)
        store.log_run(RunRecord(run_id="r1", start_time=1.0, total_cost_usd=0.5))
        result = store.export_json()
        # Should parse without error
        data = json.loads(result)
        assert isinstance(data, list)
        assert data[0]["total_cost_usd"] == 0.5
        store.close()

    def test_export_multiple_runs(self, tmp_db):
        store = TelemetryStore(tmp_db)
        for i in range(5):
            store.log_run(RunRecord(run_id=f"r{i}", start_time=float(i)))

        result = store.export_json()
        data = json.loads(result)
        assert len(data) == 5
        store.close()
