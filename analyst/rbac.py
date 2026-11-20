"""Role-based access: which tables, whether PII is visible, and row-level merchant scope."""
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


_ALL = frozenset({"transactions", "transactions_raw", "merchants", "banks",
                  "payment_methods", "settlements"})

ROLES: dict[str, Role] = {
    "head_of_payments": Role(
        "head_of_payments", "Head of Payments", _ALL, False, None,
        "All merchants, all tables. PII columns hidden."),
    "risk_ops": Role(
        "risk_ops", "Risk Ops analyst", _ALL, True, None,
        "All merchants, may read customer_vpa for investigations."),
    "kam_portfolio_a": Role(
        "kam_portfolio_a", "Key Account Manager - Portfolio A",
        frozenset({"transactions", "merchants", "banks", "payment_methods", "settlements"}),
        False, "kam_portfolio = 'PORTFOLIO_A'",
        "Only Portfolio A merchants (ecommerce + food delivery). No raw table, no PII."),
}


class AccessDenied(Exception):
    """Raised when a request falls outside the caller's access scope."""

    def __init__(self, message: str, rule: str):
        super().__init__(message)
        self.rule = rule
