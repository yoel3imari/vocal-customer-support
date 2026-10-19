"""SQLite persistence: calls, turns, per-stage latency metrics."""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger("voxdesk.store")

SCHEMA = """
CREATE TABLE IF NOT EXISTS calls (
    id TEXT PRIMARY KEY,
    tenant TEXT NOT NULL,
    model TEXT NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    duration_s REAL,
    turns INTEGER NOT NULL DEFAULT 0,
    summary TEXT
);
CREATE TABLE IF NOT EXISTS turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    call_id TEXT NOT NULL,
    role TEXT NOT NULL,
    text TEXT NOT NULL,
    ts TEXT NOT NULL,
    interrupted INTEGER NOT NULL DEFAULT 0,
    sources TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    call_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    ms REAL NOT NULL,
    ts TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_turns_call ON turns(call_id);
CREATE INDEX IF NOT EXISTS idx_metrics_call ON metrics(call_id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class CallStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def start_call(self, call_id: str, tenant: str, model: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO calls (id, tenant, model, started_at) VALUES (?, ?, ?, ?)",
                (call_id, tenant, model, _now()),
            )
            self._conn.commit()

    def add_turn(self, call_id: str, role: str, text: str, *, interrupted: bool = False,
                 sources: list[str] | None = None) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO turns (call_id, role, text, ts, interrupted, sources) VALUES (?, ?, ?, ?, ?, ?)",
                (call_id, role, text, _now(), int(interrupted), json.dumps(sources or [])),
            )
            self._conn.execute(
                "UPDATE calls SET turns = turns + 1 WHERE id = ?", (call_id,)
            )
            self._conn.commit()

    def add_metric(self, call_id: str, stage: str, ms: float) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO metrics (call_id, stage, ms, ts) VALUES (?, ?, ?, ?)",
                (call_id, stage, round(ms, 1), _now()),
            )
            self._conn.commit()

    def end_call(self, call_id: str, *, duration_s: float, summary: str | None) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE calls SET ended_at = ?, duration_s = ?, summary = ? WHERE id = ?",
                (_now(), round(duration_s, 1), summary, call_id),
            )
            self._conn.commit()

    def recent_calls(self, limit: int = 20) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, tenant, model, started_at, ended_at, duration_s, turns "
                "FROM calls ORDER BY started_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        keys = ("id", "tenant", "model", "started_at", "ended_at", "duration_s", "turns")
        return [dict(zip(keys, row)) for row in rows]

    def stats(self) -> dict:
        with self._lock:
            calls = self._conn.execute("SELECT COUNT(*), COALESCE(SUM(duration_s), 0) FROM calls").fetchone()
            turns = self._conn.execute("SELECT COUNT(*) FROM turns").fetchone()[0]
            latency = dict(
                self._conn.execute(
                    "SELECT stage, AVG(ms) FROM metrics GROUP BY stage"
                ).fetchall()
            )
        return {
            "calls": calls[0],
            "talk_seconds": round(calls[1] or 0, 1),
            "turns": turns,
            "avg_ms": {k: round(v, 1) for k, v in latency.items()},
        }

    def close(self) -> None:
        with self._lock:
            self._conn.close()
