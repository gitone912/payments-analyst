"""Read-only executor with timeout, result-size cap, caching and a persistent audit log.

Concurrency: one Executor is shared by every user of the app. Each query runs on its own DuckDB
cursor, so queries from different users run in parallel and a timeout interrupts only the query
that overran. Query ids are assigned by the caller per answer; nothing per-request is stored here
except the audit trail.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import duckdb
import pandas as pd

QUERY_TIMEOUT_S = 8.0
MAX_ROWS = 5000
MAX_CACHE_ENTRIES = 512
DEFAULT_AUDIT_PATH = os.path.join(os.path.dirname(__file__), "..", "logs", "audit.jsonl")


class QueryTimeout(Exception):
    pass


@dataclass
class QueryRecord:
    query_id: str
    purpose: str
    role: str
    sql: str
    executable_sql: str
    rows: int
    ms: float
    cached: bool
    checks: list[str] = field(default_factory=list)
    status: str = "ok"
    error: str | None = None
    user: str = ""
    at: str = ""


class Executor:
    def __init__(self, db_path: str, audit_path: str | None = None):
        self.db_path = db_path
        self._con = duckdb.connect(db_path, read_only=True)
        self._cache: dict[str, pd.DataFrame] = {}
        self._cache_lock = threading.Lock()
        self._audit_lock = threading.Lock()
        self.audit_path = audit_path if audit_path is not None else os.getenv("AUDIT_LOG", DEFAULT_AUDIT_PATH)
        if self.audit_path:
            os.makedirs(os.path.dirname(os.path.abspath(self.audit_path)), exist_ok=True)

    def raw(self, sql: str) -> pd.DataFrame:
        """Internal-only access (data-quality checks, lookups). Never used for user SQL."""
        cur = self._con.cursor()
        try:
            return cur.execute(sql).df()
        finally:
            cur.close()

    def _audit(self, rec: QueryRecord) -> None:
        """Append-only JSONL: who ran what, as which role, when, and with what outcome."""
        if not self.audit_path:
            return
        line = json.dumps({k: v for k, v in asdict(rec).items() if k != "sql"}, default=str)
        with self._audit_lock, open(self.audit_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def read_audit(self, user: str | None = None, limit: int = 200) -> list[dict]:
        """Recent audit entries, optionally only one user's (a user never sees other users' queries)."""
        if not self.audit_path or not os.path.exists(self.audit_path):
            return []
        with self._audit_lock, open(self.audit_path, encoding="utf-8") as f:
            rows = [json.loads(x) for x in f if x.strip()]
        if user is not None:
            rows = [r for r in rows if r.get("user") == user]
        return rows[-limit:]

    def run(self, executable_sql: str, *, original_sql: str, purpose: str, role: str,
            checks: list[str], query_id: str = "Q1", user: str = "") -> tuple[pd.DataFrame, QueryRecord]:
        key = hashlib.sha1(f"{role}|{executable_sql}".encode()).hexdigest()
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._cache_lock:
            hit = self._cache.get(key)
        if hit is not None:
            rec = QueryRecord(query_id, purpose, role, original_sql, executable_sql, len(hit), 0.0, True,
                              checks, user=user, at=now)
            self._audit(rec)
            return hit.copy(), rec

        cur = self._con.cursor()  # own connection handle: parallel-safe, interruptible on its own
        timer = threading.Timer(QUERY_TIMEOUT_S, cur.interrupt)
        timer.daemon = True  # never keep the process alive on shutdown
        t0 = time.perf_counter()
        try:
            timer.start()
            df = cur.execute(executable_sql).df()
        except duckdb.InterruptException:
            ms = (time.perf_counter() - t0) * 1000
            rec = QueryRecord(query_id, purpose, role, original_sql, executable_sql, 0, ms, False, checks,
                              "timeout", f"cancelled after {QUERY_TIMEOUT_S}s", user, now)
            self._audit(rec)
            raise QueryTimeout(rec.error)
        except Exception as e:  # noqa: BLE001
            ms = (time.perf_counter() - t0) * 1000
            rec = QueryRecord(query_id, purpose, role, original_sql, executable_sql, 0, ms, False, checks,
                              "error", str(e).splitlines()[0], user, now)
            self._audit(rec)
            raise
        finally:
            timer.cancel()
            cur.close()
        ms = (time.perf_counter() - t0) * 1000
        df = df.head(MAX_ROWS)
        with self._cache_lock:
            if len(self._cache) >= MAX_CACHE_ENTRIES:
                self._cache.pop(next(iter(self._cache)))  # drop the oldest entry
            self._cache[key] = df
        rec = QueryRecord(query_id, purpose, role, original_sql, executable_sql, len(df), ms, False, checks,
                          user=user, at=now)
        self._audit(rec)
        return df.copy(), rec
