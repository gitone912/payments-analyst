"""Role-based access: tables, PII, row-level merchant scope, and what each role may run or see.

Every capability here is enforced by the engine or the SQL guard; the UI only reflects it."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Role:
    key: str
    label: str
    allowed_tables: frozenset[str]
    can_see_pii: bool
    # SQL predicate on raw.merchants (alias-free) defining which merchants are visible;
    # None = all merchants
    merchant_scope_sql: str | None
    description: str
    can_run_sql: bool = False            # SQL console: hand-written (read-only, guarded) SQL
    can_view_executed_sql: bool = False  # executed SQL shows physical tables + row-level scope rewrite

    def capabilities(self) -> list[tuple[str, bool, str]]:
        """(capability, allowed, detail) for display. Mirrors what the engine enforces."""
        scope = "all merchants" if self.merchant_scope_sql is None else "own portfolio only"
        return [
            ("Ask questions", True, scope),
            ("Raw ingested data", "transactions_raw" in self.allowed_tables, "table with duplicates, as ingested"),
            ("Customer PII", self.can_see_pii, "customer VPA, for investigations"),
            ("SQL console", self.can_run_sql, "hand-written, read-only, guarded SQL"),
            ("Executed SQL", self.can_view_executed_sql, "internal tables and row-level scope"),
        ]


_ALL = frozenset({"transactions", "transactions_raw", "merchants", "banks",
                  "payment_methods", "settlements"})

ROLES: dict[str, Role] = {
    "head_of_payments": Role(
        "head_of_payments", "Head of Payments", _ALL, False, None,
        "All merchants, all tables. PII columns hidden.", can_run_sql=True),
    "risk_ops": Role(
        "risk_ops", "Risk Ops analyst", _ALL, True, None,
        "All merchants, may read customer_vpa for investigations.", can_run_sql=True,
        can_view_executed_sql=True),
    "kam_portfolio_a": Role(
        "kam_portfolio_a", "Key Account Manager - Portfolio A",
        frozenset({"transactions", "merchants", "banks", "payment_methods", "settlements"}),
        False, "kam_portfolio = 'PORTFOLIO_A'",
        "Only Portfolio A merchants (ecommerce + food delivery). No raw table, no PII.",
        can_run_sql=False, can_view_executed_sql=False),
}


class AccessDenied(Exception):
    """Raised when a request falls outside the caller's access scope."""

    def __init__(self, message: str, rule: str):
        super().__init__(message)
        self.rule = rule
