"""Tests for budgeteer.tool_executor."""

import pytest

from budgeteer.tool_executor import ToolExecutor


class TestToolExecutorRegistry:
    def test_empty_registry(self):
        executor = ToolExecutor()
        assert executor.list_tools() == []

    def test_register_and_list(self):
        executor = ToolExecutor()
        executor.register("search", lambda: None)
        executor.register("calc", lambda: None)
        assert sorted(executor.list_tools()) == ["calc", "search"]

    def test_register_overwrites(self):
        executor = ToolExecutor()
        executor.register("tool", lambda: "v1")
        executor.register("tool", lambda: "v2")
        assert len(executor.list_tools()) == 1
        result = executor.execute("tool")
        assert result.output == "v2"


class TestToolExecutorExecute:
    def test_execute_success(self):
        executor = ToolExecutor()
        executor.register("greet", lambda who: f"Hello, {who}!")
        result = executor.execute("greet", who="World")
        assert result.success is True
        assert result.output == "Hello, World!"
        assert result.tool_name == "greet"
        assert result.error is None

    def test_execute_measures_duration(self):
        executor = ToolExecutor()
        executor.register("noop", lambda: None)
        result = executor.execute("noop")
        assert result.duration_ms > 0

    def test_execute_passes_kwargs(self):
        received = {}

        def tool_fn(**kwargs):
            received.update(kwargs)
            return "ok"

        executor = ToolExecutor()
        executor.register("t", tool_fn)
        executor.execute("t", x=1, y="two")
        assert received == {"x": 1, "y": "two"}

    def test_execute_unregistered_tool(self):
        executor = ToolExecutor()
        result = executor.execute("missing_tool")
        assert result.success is False
        assert "missing_tool" in result.error
        assert result.tool_name == "missing_tool"

    def test_execute_tool_raises_exception(self):
        executor = ToolExecutor()
        executor.register("fail", lambda: (_ for _ in ()).throw(ValueError("bad input")))

        def bad_tool():
            raise ValueError("bad input")

        executor.register("fail", bad_tool)
        result = executor.execute("fail")
        assert result.success is False
        assert "bad input" in result.error

    def test_execute_tool_raises_preserves_duration(self):
        def slow_fail():
            raise RuntimeError("boom")

        executor = ToolExecutor()
        executor.register("slow_fail", slow_fail)
        result = executor.execute("slow_fail")
        assert result.success is False
        assert result.duration_ms > 0
