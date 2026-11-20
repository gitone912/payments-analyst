"""Compiles a validated Plan into SQL over the *logical* tables (the guard then scopes it)."""
from __future__ import annotations

from datetime import timedelta

from .plan import Plan, Window
from .semantic import DIMENSIONS, FILTERABLE, JOIN_SQL, METRICS


def _q(v: str) -> str:
    return "'" + str(v).replace("'", "''") + "'"


def _win(w: Window, col: str = "t.created_at") -> str:
    return f"{col} >= DATE {_q(w.start)} AND {col} < DATE {_q(w.end + timedelta(days=1))}"


def _joins(dims: list[str], filters: dict) -> str:
    needed: list[str] = []
    for d in dims:
        for j in DIMENSIONS[d].joins:
            if j not in needed:
                needed.append(j)
    for f in filters:
        tbl = {"b": "banks", "pm": "payment_methods", "m": "merchants"}[FILTERABLE[f].split(".")[0]]
        if tbl not in needed:
            needed.append(tbl)
    return "\n".join(JOIN_SQL[j] for j in needed)


def _filters(filters: dict) -> str:
    parts = []
    for k, vals in filters.items():
        col = FILTERABLE[k]
        parts.append(f"{col} IN ({', '.join(_q(v) for v in vals)})")
    return "".join(f"\n  AND {p}" for p in parts)


def fact_table(plan: Plan) -> str:
    return "transactions_raw" if plan.use_raw else "transactions"


def metric_by(plan: Plan, dims: list[str], window: Window, extra_metrics=("attempts",)) -> str:
    m = METRICS[plan.metric]
    sel_dims = [f"{DIMENSIONS[d].sql} AS {d}" for d in dims]
    sel = sel_dims + [f"{m.sql} AS {m.key}"]
    for em in extra_metrics:
        if em != m.key:
            sel.append(f"{METRICS[em].sql} AS {em}")
    group = f"\nGROUP BY {', '.join(str(i + 1) for i in range(len(dims)))}" if dims else ""
    order = f"\nORDER BY {'1' if dims and dims[0] in ('day', 'week', 'hour') else 'attempts DESC'}" if dims else ""
    return (f"SELECT {', '.join(sel)}\nFROM {fact_table(plan)} t\n{_joins(dims, plan.filters)}\n"
            f"WHERE {_win(window)}{_filters(plan.filters)}{group}{order}").replace("\n\n", "\n")


def two_period_segments(plan: Plan, dims: list[str]) -> str:
    """Numerator/denominator per segment per period - input to the mix/rate decomposition."""
    m = METRICS[plan.metric]
    num = m.numerator or m.sql
    den = m.denominator or "COUNT(*)"
    seg = " || ' · ' || ".join(f"CAST({DIMENSIONS[d].sql} AS VARCHAR)" for d in dims) or "'all'"
    w, b = plan.window, plan.baseline
    return (f"SELECT {seg} AS segment,\n"
            f"  CASE WHEN {_win(w)} THEN 'current' ELSE 'baseline' END AS period,\n"
            f"  {den} AS denom, {num} AS numer\n"
            f"FROM {fact_table(plan)} t\n{_joins(dims, plan.filters)}\n"
            f"WHERE (({_win(w)}) OR ({_win(b)})){_filters(plan.filters)}\n"
            f"GROUP BY 1, 2").replace("\n\n", "\n")


def daily_trend(plan: Plan, window: Window) -> str:
    m = METRICS[plan.metric]
    return (f"SELECT CAST(t.created_at AS DATE) AS day, {m.sql} AS {m.key}, COUNT(*) AS attempts\n"
            f"FROM {fact_table(plan)} t\n{_joins([], plan.filters)}\n"
            f"WHERE {_win(window)}{_filters(plan.filters)}\nGROUP BY 1 ORDER BY 1").replace("\n\n", "\n")


def source_reconciliation(plan: Plan) -> str:
    w = plan.window
    return f"""WITH tx AS (
  SELECT CAST(t.created_at AS DATE) AS day, COUNT(*) AS tx_attempts,
         SUM(CASE WHEN t.status = 'SUCCESS' THEN 1 ELSE 0 END) AS tx_success,
         SUM(CASE WHEN t.status = 'PENDING' THEN 1 ELSE 0 END) AS tx_pending
  FROM transactions t
  WHERE {_win(w)}
  GROUP BY 1
), st AS (
  SELECT s.settlement_date AS day, SUM(s.attempted_count) AS st_attempts, SUM(s.success_count) AS st_success
  FROM settlements s
  WHERE {_win(w, 's.settlement_date')}
  GROUP BY 1
)
SELECT tx.day, tx_attempts, st_attempts, tx_success, st_success, tx_pending,
       tx_success * 1.0 / tx_attempts AS tx_success_rate,
       st_success * 1.0 / st_attempts AS st_success_rate
FROM tx JOIN st ON tx.day = st.day
ORDER BY tx.day"""
