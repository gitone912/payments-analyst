"""
Deterministic fixes for a known LLM SQL mistake, applied before execution and always disclosed.

`created_at <= '2026-10-25'` compares a TIMESTAMP with midnight at the START of 25 Oct, so the whole
last day is silently dropped (a 'last week' answer becomes six days). Same for
`created_at BETWEEN '2026-10-19' AND '2026-10-25'`. We rewrite the upper bound to
`< '2026-10-26'`, which is what the question meant.
"""
from __future__ import annotations

import re
from datetime import date, timedelta

import sqlglot
from sqlglot import exp

TIMESTAMP_COLUMNS = {"created_at", "ingested_at"}
_DATE_ONLY = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _is_ts_col(node) -> bool:
    return isinstance(node, exp.Column) and node.name.lower() in TIMESTAMP_COLUMNS


def _date_literal(node) -> date | None:
    if isinstance(node, exp.Literal) and node.is_string and _DATE_ONLY.match(node.this):
        return date.fromisoformat(node.this)
    return None


def fix_inclusive_end_dates(sql: str) -> tuple[str, list[str]]:
    """Returns (sql, notes). sql is unchanged and notes empty when nothing needed fixing."""
    try:
        tree = sqlglot.parse_one(sql, read="duckdb")
    except sqlglot.errors.ParseError:
        return sql, []
    notes: list[str] = []

    def fix(node):
        if isinstance(node, exp.LTE) and _is_ts_col(node.this) and _date_literal(node.expression):
            d = _date_literal(node.expression)
            new = exp.LT(this=node.this.copy(), expression=exp.Literal.string(str(d + timedelta(days=1))))
            notes.append(f"`{node.sql()}` → `{new.sql()}`")
            return new
        if isinstance(node, exp.Between) and _is_ts_col(node.this) and _date_literal(node.args.get("high")):
            d = _date_literal(node.args["high"])
            new = exp.And(this=exp.GTE(this=node.this.copy(), expression=node.args["low"].copy()),
                          expression=exp.LT(this=node.this.copy(),
                                            expression=exp.Literal.string(str(d + timedelta(days=1)))))
            notes.append(f"`{node.sql()}` → `{new.sql()}`")
            return exp.Paren(this=new)
        return node

    fixed = tree.transform(fix)
    if not notes:
        return sql, []
    return fixed.sql(dialect="duckdb"), notes
