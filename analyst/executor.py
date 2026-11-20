"""Read-only executor with timeout, result-size cap, caching and an audit log."""
from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass, field

import duckdb
import pandas as pd

QUERY_TIMEOUT_S = 8.0
MAX_ROWS = 5000


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


class Executor:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._con = duckdb.connect(db_path, read_only=True)
        self._cache: dict[str, pd.DataFrame] = {}
        self._lock = threading.Lock()
        self.log: list[QueryRecord] = []

    def raw(self, sql: str) -> pd.DataFrame:
        """Internal-only access (data-quality checks, lookups). Never used for user SQL."""
        with self._lock:
            return self._con.execute(sql).df()

    def run(self, executable_sql: str, *, original_sql: str, purpose: str, role: str,
            checks: list[str]) -> tuple[pd.DataFrame, QueryRecord]:
        key = hashlib.sha1(f"{role}|{executable_sql}".encode()).hexdigest()[:10]
        qid = f"Q{len(self.log) + 1}"
        if key in self._cache:
            df = self._cache[key]
            rec = QueryRecord(qid, purpose, role, original_sql, executable_sql, len(df), 0.0, True, checks)
            self.log.append(rec)
            return df.copy(), rec

        timer = threading.Timer(QUERY_TIMEOUT_S, self._con.interrupt)
        t0 = time.perf_counter()
        try:
            with self._lock:
                timer.start()
                df = self._con.execute(executable_sql).df()
        except duckdb.InterruptException:
            ms = (time.perf_counter() - t0) * 1000
            rec = QueryRecord(qid, purpose, role, original_sql, executable_sql, 0, ms, False, checks,
                              "timeout", f"cancelled after {QUERY_TIMEOUT_S}s")
            self.log.append(rec)
            raise QueryTimeout(rec.error)
        except Exception as e:  # noqa: BLE001
            ms = (time.perf_counter() - t0) * 1000
            rec = QueryRecord(qid, purpose, role, original_sql, executable_sql, 0, ms, False, checks,
                              "error", str(e).splitlines()[0])
            self.log.append(rec)
            raise
        finally:
            timer.cancel()
        ms = (time.perf_counter() - t0) * 1000
        df = df.head(MAX_ROWS)
        self._cache[key] = df
        rec = QueryRecord(qid, purpose, role, original_sql, executable_sql, len(df), ms, False, checks)
        self.log.append(rec)
        return df.copy(), rec
