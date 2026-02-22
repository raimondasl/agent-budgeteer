"""Tests for budgeteer.telemetry — including budget ledger."""

from budgeteer.models import RunRecord, StepDecision, StepMetrics, StepRecord, ToolRecord


class TestBudgetLedger:
    def test_empty_usage(self, telemetry):
        usage = telemetry.get_daily_usage("user", "u1")
        assert usage == {"cost_usd": 0.0, "tokens": 0, "runs": 0}

    def test_record_and_get(self, telemetry):
        telemetry.record_daily_usage(
            "user", "u1", cost_usd=1.5, tokens=500, runs=1, date="2025-01-15"
        )
        usage = telemetry.get_daily_usage("user", "u1", date="2025-01-15")
        assert usage["cost_usd"] == 1.5
        assert usage["tokens"] == 500
        assert usage["runs"] == 1

    def test_incremental_updates(self, telemetry):
        telemetry.record_daily_usage(
            "org", "org1", cost_usd=2.0, tokens=1000, date="2025-01-15"
        )
        telemetry.record_daily_usage(
            "org", "org1", cost_usd=3.0, tokens=2000, date="2025-01-15"
        )
        usage = telemetry.get_daily_usage("org", "org1", date="2025-01-15")
        assert usage["cost_usd"] == 5.0
        assert usage["tokens"] == 3000

    def test_different_dates_isolated(self, telemetry):
        telemetry.record_daily_usage(
            "user", "u1", cost_usd=10.0, date="2025-01-14"
        )
        telemetry.record_daily_usage(
            "user", "u1", cost_usd=5.0, date="2025-01-15"
        )
        assert telemetry.get_daily_usage("user", "u1", date="2025-01-14")["cost_usd"] == 10.0
        assert telemetry.get_daily_usage("user", "u1", date="2025-01-15")["cost_usd"] == 5.0

    def test_different_scopes_isolated(self, telemetry):
        telemetry.record_daily_usage(
            "user", "u1", cost_usd=1.0, date="2025-01-15"
        )
        telemetry.record_daily_usage(
            "user", "u2", cost_usd=2.0, date="2025-01-15"
        )
        assert telemetry.get_daily_usage("user", "u1", date="2025-01-15")["cost_usd"] == 1.0
        assert telemetry.get_daily_usage("user", "u2", date="2025-01-15")["cost_usd"] == 2.0


class TestTelemetryRunPersistence:
    def test_log_and_get_run(self, telemetry):
        run = RunRecord(run_id="r1")
        telemetry.log_run(run)
        retrieved = telemetry.get_run("r1")
        assert retrieved.run_id == "r1"
        assert retrieved.total_cost_usd == 0.0

    def test_update_run(self, telemetry):
        run = RunRecord(run_id="r1")
        telemetry.log_run(run)
        run.total_cost_usd = 1.5
        run.total_tokens = 1000
        run.success = True
        telemetry.update_run(run)

        retrieved = telemetry.get_run("r1")
        assert retrieved.total_cost_usd == 1.5
        assert retrieved.total_tokens == 1000
        assert retrieved.success is True


class TestTelemetryStepPersistence:
    def test_log_and_get_steps(self, telemetry):
        run = RunRecord(run_id="r1")
        telemetry.log_run(run)

        decision = StepDecision(model="gpt-4o")
        metrics = StepMetrics(prompt_tokens=100, cost_usd=0.01)
        step = StepRecord(
            run_id="r1", step_id="s1", decision=decision, actual=metrics
        )
        telemetry.log_step(step)

        steps = telemetry.get_steps("r1")
        assert len(steps) == 1
        assert steps[0].decision.model == "gpt-4o"
        assert steps[0].actual.prompt_tokens == 100


class TestTelemetryToolCallPersistence:
    def test_log_and_get_tool_calls(self, telemetry):
        run = RunRecord(run_id="r1")
        telemetry.log_run(run)

        record = ToolRecord(
            run_id="r1", step_id="s1", tool_name="search", duration_ms=150.0
        )
        telemetry.log_tool_call(record)

        calls = telemetry.get_tool_calls("r1")
        assert len(calls) == 1
        assert calls[0].tool_name == "search"

    def test_filter_by_step(self, telemetry):
        run = RunRecord(run_id="r1")
        telemetry.log_run(run)

        telemetry.log_tool_call(
            ToolRecord(run_id="r1", step_id="s1", tool_name="a", duration_ms=10)
        )
        telemetry.log_tool_call(
            ToolRecord(run_id="r1", step_id="s2", tool_name="b", duration_ms=20)
        )

        calls = telemetry.get_tool_calls("r1", step_id="s1")
        assert len(calls) == 1
        assert calls[0].tool_name == "a"


class TestTelemetryEdgeCases:
    def test_get_steps_nonexistent_run(self, telemetry):
        assert telemetry.get_steps("nonexistent") == []

    def test_get_tool_calls_nonexistent_run(self, telemetry):
        assert telemetry.get_tool_calls("nonexistent") == []

    def test_get_run_nonexistent(self, telemetry):
        assert telemetry.get_run("nonexistent") is None

    def test_get_run_summary_with_no_steps_or_tools(self, telemetry):
        run = RunRecord(run_id="empty-run")
        telemetry.log_run(run)
        summary = telemetry.get_run_summary("empty-run")
        assert summary is not None
        assert summary["logged_steps"] == 0
        assert summary["logged_tool_calls"] == 0
        assert summary["run_id"] == "empty-run"
