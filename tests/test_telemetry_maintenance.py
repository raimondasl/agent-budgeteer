"""Tests for telemetry maintenance: purge, stats, schema versioning (Milestone 16)."""

from __future__ import annotations

import time

import pytest

from budgeteer.config import BudgeteerConfig
from budgeteer.models import RunRecord, StepDecision, StepMetrics, StepRecord, ToolRecord
from budgeteer.sdk import Budgeteer
from budgeteer.telemetry import TelemetryStore, _SCHEMA_VERSION


def _make_store(tmp_path):
    return TelemetryStore(str(tmp_path / "test.db"))


class TestSchemaVersioning:
    """Tests for schema_versions table."""

    def test_schema_version_recorded(self, tmp_path):
        store = _make_store(tmp_path)
        assert store.get_schema_version() == _SCHEMA_VERSION
        store.close()

    def test_schema_version_is_1(self, tmp_path):
        store = _make_store(tmp_path)
        assert store.get_schema_version() == 1
        store.close()

    def test_reopen_does_not_duplicate_version(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        s1 = TelemetryStore(db_path)
        s1.close()
        s2 = TelemetryStore(db_path)
        assert s2.get_schema_version() == 1
        s2.close()


class TestPurgeBefore:
    """Tests for purge_before()."""

    def test_purge_old_runs(self, tmp_path):
        store = _make_store(tmp_path)
        old = RunRecord(run_id="old", start_time=1000.0)
        new = RunRecord(run_id="new", start_time=time.time())
        store.log_run(old)
        store.log_run(new)

        deleted = store.purge_before(2000.0)
        assert deleted == 1
        assert store.get_run("old") is None
        assert store.get_run("new") is not None
        store.close()

    def test_purge_cascades_to_steps(self, tmp_path):
        store = _make_store(tmp_path)
        run = RunRecord(run_id="r1", start_time=1000.0)
        store.log_run(run)
        step = StepRecord(run_id="r1", step_id="s1", decision=StepDecision(model="m"),
                          actual=StepMetrics(cost_usd=0.01))
        store.log_step(step)

        store.purge_before(2000.0)
        assert store.get_steps("r1") == []
        store.close()

    def test_purge_cascades_to_tool_calls(self, tmp_path):
        store = _make_store(tmp_path)
        run = RunRecord(run_id="r1", start_time=1000.0)
        store.log_run(run)
        tool = ToolRecord(run_id="r1", step_id="s1", tool_name="t1", duration_ms=10)
        store.log_tool_call(tool)

        store.purge_before(2000.0)
        assert store.get_tool_calls("r1") == []
        store.close()

    def test_purge_preserves_recent(self, tmp_path):
        store = _make_store(tmp_path)
        now = time.time()
        for i in range(5):
            store.log_run(RunRecord(run_id=f"r{i}", start_time=now + i))

        deleted = store.purge_before(now - 100)
        assert deleted == 0
        assert len(store.list_run_ids()) == 5
        store.close()

    def test_purge_returns_count(self, tmp_path):
        store = _make_store(tmp_path)
        for i in range(3):
            store.log_run(RunRecord(run_id=f"r{i}", start_time=1000.0 + i))
        store.log_run(RunRecord(run_id="recent", start_time=time.time()))

        deleted = store.purge_before(5000.0)
        assert deleted == 3
        store.close()

    def test_purge_empty_db(self, tmp_path):
        store = _make_store(tmp_path)
        assert store.purge_before(time.time()) == 0
        store.close()


class TestPurgeRun:
    """Tests for purge_run()."""

    def test_purge_specific_run(self, tmp_path):
        store = _make_store(tmp_path)
        store.log_run(RunRecord(run_id="r1"))
        store.log_run(RunRecord(run_id="r2"))
        store.log_step(StepRecord(run_id="r1", step_id="s1", decision=StepDecision(model="m")))

        store.purge_run("r1")
        assert store.get_run("r1") is None
        assert store.get_steps("r1") == []
        assert store.get_run("r2") is not None
        store.close()

    def test_purge_nonexistent_run(self, tmp_path):
        store = _make_store(tmp_path)
        store.purge_run("nonexistent")  # should not raise
        store.close()


class TestGetStats:
    """Tests for get_stats()."""

    def test_empty_stats(self, tmp_path):
        store = _make_store(tmp_path)
        stats = store.get_stats()
        assert stats["runs"] == 0
        assert stats["steps"] == 0
        assert stats["tool_calls"] == 0
        assert stats["budget_ledger"] == 0
        assert stats["db_size_bytes"] > 0
        store.close()

    def test_stats_after_data(self, tmp_path):
        store = _make_store(tmp_path)
        store.log_run(RunRecord(run_id="r1"))
        store.log_step(StepRecord(run_id="r1", step_id="s1", decision=StepDecision(model="m")))
        store.log_step(StepRecord(run_id="r1", step_id="s2", decision=StepDecision(model="m")))
        store.log_tool_call(ToolRecord(run_id="r1", step_id="s1", tool_name="t", duration_ms=10))
        stats = store.get_stats()
        assert stats["runs"] == 1
        assert stats["steps"] == 2
        assert stats["tool_calls"] == 1
        store.close()

    def test_stats_reflect_purge(self, tmp_path):
        store = _make_store(tmp_path)
        store.log_run(RunRecord(run_id="r1", start_time=1000.0))
        store.log_step(StepRecord(run_id="r1", step_id="s1", decision=StepDecision(model="m")))
        store.purge_before(2000.0)
        stats = store.get_stats()
        assert stats["runs"] == 0
        assert stats["steps"] == 0
        store.close()


class TestRetentionAutoPurge:
    """Tests for retention_days auto-purge via SDK."""

    def test_retention_purges_old_on_init(self, tmp_path):
        db_path = str(tmp_path / "tel.db")
        # First: create some old data
        store = TelemetryStore(db_path)
        old_time = time.time() - 100 * 86400  # 100 days ago
        store.log_run(RunRecord(run_id="old", start_time=old_time))
        store.log_run(RunRecord(run_id="recent", start_time=time.time()))
        store.close()

        # Second: init SDK with retention_days=30
        cfg = BudgeteerConfig(storage_path=db_path, retention_days=30)
        b = Budgeteer(config=cfg)
        assert b.telemetry.get_run("old") is None
        assert b.telemetry.get_run("recent") is not None
        b.close()

    def test_no_retention_preserves_all(self, tmp_path):
        db_path = str(tmp_path / "tel.db")
        store = TelemetryStore(db_path)
        old_time = time.time() - 365 * 86400  # 1 year ago
        store.log_run(RunRecord(run_id="ancient", start_time=old_time))
        store.close()

        cfg = BudgeteerConfig(storage_path=db_path)  # no retention_days
        b = Budgeteer(config=cfg)
        assert b.telemetry.get_run("ancient") is not None
        b.close()
