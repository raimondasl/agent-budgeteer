"""SQLite-backed telemetry storage for Budgeteer.

Stores run records, step records, and tool call records for analysis
and calibration.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import asdict
from pathlib import Path

from budgeteer.models import (
    RunRecord,
    StepDecision,
    StepMetrics,
    StepRecord,
    ToolRecord,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    start_time REAL NOT NULL,
    end_time REAL,
    total_cost_usd REAL DEFAULT 0,
    total_tokens INTEGER DEFAULT 0,
    total_latency_ms REAL DEFAULT 0,
    total_steps INTEGER DEFAULT 0,
    total_tool_calls INTEGER DEFAULT 0,
    success INTEGER
);

CREATE TABLE IF NOT EXISTS steps (
    run_id TEXT NOT NULL,
    step_id TEXT NOT NULL,
    decision_json TEXT NOT NULL,
    predicted_json TEXT,
    actual_json TEXT,
    timestamp REAL NOT NULL,
    PRIMARY KEY (run_id, step_id),
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE TABLE IF NOT EXISTS tool_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    step_id TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    duration_ms REAL NOT NULL,
    tokens_used INTEGER DEFAULT 0,
    success INTEGER DEFAULT 1,
    error TEXT,
    timestamp REAL NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE TABLE IF NOT EXISTS budget_ledger (
    scope TEXT NOT NULL,
    scope_id TEXT NOT NULL,
    date TEXT NOT NULL,
    total_cost_usd REAL DEFAULT 0,
    total_tokens INTEGER DEFAULT 0,
    total_runs INTEGER DEFAULT 0,
    PRIMARY KEY (scope, scope_id, date)
);
"""


def _bool_to_int(val: bool | None) -> int | None:
    if val is None:
        return None
    return 1 if val else 0


def _int_to_bool(val: int | None) -> bool | None:
    if val is None:
        return None
    return bool(val)


class TelemetryStore:
    """SQLite-backed storage for telemetry records."""

    def __init__(self, db_path: str | Path = "budgeteer_telemetry.db"):
        self._db_path = str(db_path)
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    # -- Runs --

    def log_run(self, record: RunRecord) -> None:
        """Insert a new run record."""
        with self._lock:
            self._conn.execute(
                """INSERT INTO runs
                   (run_id, start_time, end_time, total_cost_usd, total_tokens,
                    total_latency_ms, total_steps, total_tool_calls, success)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    record.run_id,
                    record.start_time,
                    record.end_time,
                    record.total_cost_usd,
                    record.total_tokens,
                    record.total_latency_ms,
                    record.total_steps,
                    record.total_tool_calls,
                    _bool_to_int(record.success),
                ),
            )
            self._conn.commit()

    def update_run(self, record: RunRecord) -> None:
        """Update an existing run record."""
        with self._lock:
            self._conn.execute(
                """UPDATE runs SET
                   end_time=?, total_cost_usd=?, total_tokens=?,
                   total_latency_ms=?, total_steps=?, total_tool_calls=?, success=?
                   WHERE run_id=?""",
                (
                    record.end_time,
                    record.total_cost_usd,
                    record.total_tokens,
                    record.total_latency_ms,
                    record.total_steps,
                    record.total_tool_calls,
                    _bool_to_int(record.success),
                    record.run_id,
                ),
            )
            self._conn.commit()

    def get_run(self, run_id: str) -> RunRecord | None:
        """Retrieve a run record by ID."""
        row = self._conn.execute(
            "SELECT * FROM runs WHERE run_id=?", (run_id,)
        ).fetchone()
        if row is None:
            return None
        return RunRecord(
            run_id=row["run_id"],
            start_time=row["start_time"],
            end_time=row["end_time"],
            total_cost_usd=row["total_cost_usd"],
            total_tokens=row["total_tokens"],
            total_latency_ms=row["total_latency_ms"],
            total_steps=row["total_steps"],
            total_tool_calls=row["total_tool_calls"],
            success=_int_to_bool(row["success"]),
        )

    # -- Steps --

    def log_step(self, record: StepRecord) -> None:
        """Insert a step record."""
        with self._lock:
            self._conn.execute(
                """INSERT INTO steps
                   (run_id, step_id, decision_json, predicted_json, actual_json, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    record.run_id,
                    record.step_id,
                    json.dumps(asdict(record.decision)),
                    json.dumps(asdict(record.predicted)) if record.predicted else None,
                    json.dumps(asdict(record.actual)) if record.actual else None,
                    record.timestamp,
                ),
            )
            self._conn.commit()

    def get_steps(self, run_id: str) -> list[StepRecord]:
        """Retrieve all step records for a run."""
        rows = self._conn.execute(
            "SELECT * FROM steps WHERE run_id=? ORDER BY timestamp", (run_id,)
        ).fetchall()
        results = []
        for row in rows:
            decision = StepDecision(**json.loads(row["decision_json"]))
            predicted_raw = row["predicted_json"]
            actual_raw = row["actual_json"]
            results.append(
                StepRecord(
                    run_id=row["run_id"],
                    step_id=row["step_id"],
                    decision=decision,
                    predicted=StepMetrics(**json.loads(predicted_raw)) if predicted_raw else None,
                    actual=StepMetrics(**json.loads(actual_raw)) if actual_raw else None,
                    timestamp=row["timestamp"],
                )
            )
        return results

    # -- Tool calls --

    def log_tool_call(self, record: ToolRecord) -> None:
        """Insert a tool call record."""
        with self._lock:
            self._conn.execute(
                """INSERT INTO tool_calls
                   (run_id, step_id, tool_name, duration_ms, tokens_used, success, error, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    record.run_id,
                    record.step_id,
                    record.tool_name,
                    record.duration_ms,
                    record.tokens_used,
                    _bool_to_int(record.success),
                    record.error,
                    record.timestamp,
                ),
            )
            self._conn.commit()

    def get_tool_calls(self, run_id: str, step_id: str | None = None) -> list[ToolRecord]:
        """Retrieve tool call records, optionally filtered by step."""
        if step_id:
            rows = self._conn.execute(
                "SELECT * FROM tool_calls WHERE run_id=? AND step_id=? ORDER BY timestamp",
                (run_id, step_id),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM tool_calls WHERE run_id=? ORDER BY timestamp",
                (run_id,),
            ).fetchall()
        return [
            ToolRecord(
                run_id=row["run_id"],
                step_id=row["step_id"],
                tool_name=row["tool_name"],
                duration_ms=row["duration_ms"],
                tokens_used=row["tokens_used"],
                success=_int_to_bool(row["success"]),
                error=row["error"],
                timestamp=row["timestamp"],
            )
            for row in rows
        ]

    # -- Budget ledger --

    def record_daily_usage(
        self,
        scope: str,
        scope_id: str,
        cost_usd: float = 0.0,
        tokens: int = 0,
        runs: int = 0,
        date: str | None = None,
    ) -> None:
        """Atomically increment daily usage counters for a budget scope.

        If no row exists for the given date, one is created.
        """
        if date is None:
            from datetime import datetime, timezone

            date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        with self._lock:
            self._conn.execute(
                """INSERT INTO budget_ledger (scope, scope_id, date, total_cost_usd, total_tokens, total_runs)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(scope, scope_id, date)
                   DO UPDATE SET
                       total_cost_usd = total_cost_usd + excluded.total_cost_usd,
                       total_tokens = total_tokens + excluded.total_tokens,
                       total_runs = total_runs + excluded.total_runs""",
                (scope, scope_id, date, cost_usd, tokens, runs),
            )
            self._conn.commit()

    def get_daily_usage(
        self, scope: str, scope_id: str, date: str | None = None
    ) -> dict:
        """Return daily usage for a budget scope.

        Returns a dict with keys: cost_usd, tokens, runs.
        If no data exists for the date, returns zeros.
        """
        if date is None:
            from datetime import datetime, timezone

            date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        row = self._conn.execute(
            "SELECT total_cost_usd, total_tokens, total_runs FROM budget_ledger "
            "WHERE scope=? AND scope_id=? AND date=?",
            (scope, scope_id, date),
        ).fetchone()
        if row is None:
            return {"cost_usd": 0.0, "tokens": 0, "runs": 0}
        return {
            "cost_usd": row["total_cost_usd"],
            "tokens": row["total_tokens"],
            "runs": row["total_runs"],
        }

    def list_run_ids(self) -> list[str]:
        """Return all run IDs in the store, ordered by start_time."""
        rows = self._conn.execute(
            "SELECT run_id FROM runs ORDER BY start_time"
        ).fetchall()
        return [row["run_id"] for row in rows]

    def get_run_summary(self, run_id: str) -> dict | None:
        """Get a summary of a run including step and tool call counts."""
        run = self.get_run(run_id)
        if run is None:
            return None
        step_count = self._conn.execute(
            "SELECT COUNT(*) FROM steps WHERE run_id=?", (run_id,)
        ).fetchone()[0]
        tool_count = self._conn.execute(
            "SELECT COUNT(*) FROM tool_calls WHERE run_id=?", (run_id,)
        ).fetchone()[0]
        return {
            "run_id": run.run_id,
            "start_time": run.start_time,
            "end_time": run.end_time,
            "total_cost_usd": run.total_cost_usd,
            "total_tokens": run.total_tokens,
            "total_latency_ms": run.total_latency_ms,
            "total_steps": run.total_steps,
            "total_tool_calls": run.total_tool_calls,
            "success": run.success,
            "logged_steps": step_count,
            "logged_tool_calls": tool_count,
        }

    def export_json(self, run_ids: list[str] | None = None) -> str:
        """Export run data as a JSON string.

        Args:
            run_ids: Run IDs to export. If None, exports all runs.

        Returns a JSON string containing runs, steps, and tool calls.
        """
        if run_ids is None:
            run_ids = self.list_run_ids()

        data: list[dict] = []
        for run_id in run_ids:
            run = self.get_run(run_id)
            if run is None:
                continue
            steps = self.get_steps(run_id)
            tools = self.get_tool_calls(run_id)
            data.append({
                "run_id": run.run_id,
                "start_time": run.start_time,
                "end_time": run.end_time,
                "total_cost_usd": run.total_cost_usd,
                "total_tokens": run.total_tokens,
                "total_latency_ms": run.total_latency_ms,
                "total_steps": run.total_steps,
                "total_tool_calls": run.total_tool_calls,
                "success": run.success,
                "steps": [
                    {
                        "step_id": s.step_id,
                        "decision": asdict(s.decision),
                        "predicted": asdict(s.predicted) if s.predicted else None,
                        "actual": asdict(s.actual) if s.actual else None,
                        "timestamp": s.timestamp,
                    }
                    for s in steps
                ],
                "tool_calls": [
                    {
                        "tool_name": t.tool_name,
                        "duration_ms": t.duration_ms,
                        "tokens_used": t.tokens_used,
                        "success": t.success,
                        "error": t.error,
                        "timestamp": t.timestamp,
                    }
                    for t in tools
                ],
            })
        return json.dumps(data, indent=2)
