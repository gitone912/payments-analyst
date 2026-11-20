"""
SQL guard. Every query - compiled from a plan OR written free-form by the LLM - goes
through here before it touches the database.

Checks, in order:
  1. Parses as exactly one statement, and that statement is a read-only SELECT.
  2. Only allow-listed functions (pure aggregate/date/math/string); no table functions.
  3. Only logical tables from the semantic layer; no schema-qualified access.
  4. Role checks: table allowlist, PII columns, out-of-scope merchant literals.
  5. Cost heuristics: no cartesian joins, bounded fact-table scans, bounded joins.
  6. Rewrites logical tables -> physical sources with row-level merchant scope applied,
     and adds a LIMIT if missing.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import sqlglot
from sqlglot import exp

from .rbac import AccessDenied, Role
from .semantic import FACT_TABLES, LOGICAL_TABLES, PII_COLUMNS

# Allowlist, not denylist: a function is rejected unless it is known to be a pure computation.
# DuckDB has hundreds of functions and extensions (file readers, HTTP, settings, system calls);
# a denylist misses the one nobody thought of, an allowlist fails closed.
ALLOWED_FUNCTIONS = frozenset("""
    Count Sum Avg Min Max Stddev StddevPop StddevSamp Variance VariancePop Median Quantile
    PercentileCont PercentileDisc ApproxQuantile ApproxDistinct CountIf ArgMax ArgMin AnyValue Mode
    And Or Not Xor Case If Coalesce Nullif Greatest Least Cast TryCast Paren
    Extract DateTrunc TimestampTrunc Date CurrentDate DateAdd DateSub DateDiff TimestampAdd
    TimestampSub TimestampDiff DayOfWeek DayOfWeekIso DayOfMonth DayOfYear WeekOfYear Hour Minute
    Month Year Day Week Quarter StrToTime TimeToStr StrToDate TsOrDsToDate TimeStrToTime TimeToUnix
    Abs Round Floor Ceil Sqrt Pow Ln Log Exp Sign SafeDivide Div
    Lower Upper Length Substring Concat ConcatWs Trim Pad Left Right StrPosition Initcap
    RowNumber Rank DenseRank PercentRank CumeDist Ntile Lag Lead FirstValue LastValue
""".split())
# DuckDB functions sqlglot does not model as typed expressions (parsed as Anonymous)
ALLOWED_ANONYMOUS = frozenset("""
    date_part datepart isodow dow yearweek dayname monthname epoch_ms round_even bool_or bool_and
    quantile_cont quantile_disc
""".split())
MAX_JOINS = 5
MAX_FACT_SCANS = 2
DEFAULT_LIMIT = 1000


class UnsafeQuery(Exception):
    def __init__(self, message: str, rule: str):
        super().__init__(message)
        self.rule = rule


@dataclass
class GuardResult:
    original_sql: str
    executable_sql: str
    tables: list[str]
    checks: list[str] = field(default_factory=list)   # human-readable passed checks
    cost_note: str = ""


def _fail(msg: str, rule: str):
    raise UnsafeQuery(msg, rule)


class SqlGuard:
    def __init__(self, merchant_lookup: dict[str, dict]):
        """merchant_lookup: merchant_name/merchant_id (lowercase) -> {"id","name","portfolio"}"""
        self.merchant_lookup = merchant_lookup

    # ------------------------------------------------------------------ public
    def check(self, sql: str, role: Role, visible_merchant_ids: set[str] | None) -> GuardResult:
        checks: list[str] = []
        try:
            statements = sqlglot.parse(sql, read="duckdb")
        except sqlglot.errors.ParseError as e:
            _fail(f"SQL does not parse: {str(e).splitlines()[0]}", "parse")
        statements = [s for s in statements if s is not None]
        if len(statements) != 1:
            _fail(f"Expected exactly one statement, got {len(statements)}.", "single_statement")
        tree = statements[0]
        checks.append("parses as a single statement")

        if not isinstance(tree, (exp.Select, exp.Union, exp.Intersect, exp.Except)):
            _fail(f"Only SELECT queries are allowed (got {tree.key.upper()}).", "read_only")
        for bad in (exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create, exp.Alter,
                    exp.Command, exp.Merge, exp.Into, exp.Copy):
            if tree.find(bad):
                _fail(f"Statement contains a forbidden {bad.__name__.upper()} clause.", "read_only")
        checks.append("read-only SELECT")

        for fn in tree.find_all(exp.Func):
            if isinstance(fn, exp.Anonymous):
                ok, name = fn.name.lower() in ALLOWED_ANONYMOUS, fn.name.lower()
            else:
                ok, name = type(fn).__name__ in ALLOWED_FUNCTIONS, fn.sql_name().lower()
            if not ok:
                _fail(f"Function '{name}' is not on the allowed list (aggregates, date, math and string "
                      "functions only).", "function_not_allowed")
        checks.append("only allow-listed functions")

        cte_names = {c.alias_or_name.lower() for c in tree.find_all(exp.CTE)}
        tables: list[exp.Table] = []
        for t in tree.find_all(exp.Table):
            nm = t.name.lower()
            if nm in cte_names and not t.db:
                continue
            if not nm:
                _fail("Table functions / subscripted sources are not allowed.", "table_function")
            if t.db or t.catalog:
                _fail(f"Schema-qualified table '{t.sql()}' is not allowed; use logical tables.",
                      "schema_access")
            if nm not in LOGICAL_TABLES:
                _fail(f"Unknown table '{nm}'. Allowed: {', '.join(sorted(LOGICAL_TABLES))}.",
                      "unknown_table")
            if nm not in role.allowed_tables:
                raise AccessDenied(f"Role '{role.label}' may not query table '{nm}'.", "table_scope")
            tables.append(t)
        checks.append("only catalog tables: " + ", ".join(sorted({t.name.lower() for t in tables})))

        # PII
        if not role.can_see_pii:
            for col in tree.find_all(exp.Column):
                if col.name.lower() in PII_COLUMNS:
                    raise AccessDenied(
                        f"Column '{col.name}' is PII and is not visible to role '{role.label}'. "
                        "Ask Risk Ops if this is needed for an investigation.", "pii")
            if any(t.name.lower() in FACT_TABLES for t in tables):
                for star in tree.find_all(exp.Star):
                    parent = star.parent
                    if isinstance(parent, (exp.Count,)):
                        continue
                    raise AccessDenied(
                        "SELECT * on transactions would expose PII columns; list columns explicitly.",
                        "pii")
            checks.append("no PII columns referenced")

        # merchant literal scope
        if visible_merchant_ids is not None:
            for lit in tree.find_all(exp.Literal):
                if not lit.is_string:
                    continue
                key = lit.this.lower()
                m = self.merchant_lookup.get(key)
                if m and m["id"] not in visible_merchant_ids:
                    raise AccessDenied(
                        f"Merchant '{m['name']}' is outside your portfolio ({role.label}).",
                        "row_scope")
            checks.append("merchant filters within role scope")

        # cost heuristics
        joins = list(tree.find_all(exp.Join))
        for j in joins:
            kind = (j.args.get("kind") or "").upper()
            if kind == "CROSS" or (not j.args.get("on") and not j.args.get("using")):
                _fail("Join without a join condition (cartesian product) - rejected as too expensive.",
                      "cartesian_join")
        if len(joins) > MAX_JOINS:
            _fail(f"Query has {len(joins)} joins (max {MAX_JOINS}).", "too_many_joins")
        fact_scans = sum(1 for t in tables if t.name.lower() in FACT_TABLES)
        if fact_scans > MAX_FACT_SCANS:
            _fail(f"Query scans the transactions fact table {fact_scans} times (max {MAX_FACT_SCANS}).",
                  "fact_scans")
        cost_note = f"{fact_scans} fact-table scan(s), {len(joins)} join(s)"
        checks.append("cost heuristics ok (" + cost_note + ")")

        # rewrite tables -> physical with row-level scope
        scope_ids = None if visible_merchant_ids is None else sorted(visible_merchant_ids)

        def _physical(nm: str) -> str:
            src = LOGICAL_TABLES[nm]
            if scope_ids is None or nm in ("banks", "payment_methods"):
                return f"SELECT * FROM {src}"
            ids = ", ".join(f"'{i}'" for i in scope_ids)
            return f"SELECT * FROM {src} WHERE merchant_id IN ({ids})"

        def _transform(node):
            if isinstance(node, exp.Table) and node.name.lower() in LOGICAL_TABLES \
                    and not node.db and not (node.name.lower() in cte_names):
                alias = node.alias or node.name
                sub = sqlglot.parse_one(_physical(node.name.lower()), read="duckdb")
                return exp.Subquery(this=sub, alias=exp.TableAlias(this=exp.to_identifier(alias)))
            return node

        rewritten = tree.transform(_transform)
        if isinstance(rewritten, exp.Select) and not rewritten.args.get("limit"):
            rewritten = rewritten.limit(DEFAULT_LIMIT)
            checks.append(f"LIMIT {DEFAULT_LIMIT} added")
        if scope_ids is not None:
            checks.append(f"row-level scope applied ({len(scope_ids)} merchants)")

        return GuardResult(
            original_sql=sql.strip(),
            executable_sql=rewritten.sql(dialect="duckdb"),
            tables=sorted({t.name.lower() for t in tables}),
            checks=checks,
            cost_note=cost_note,
        )
