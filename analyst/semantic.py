"""
Semantic layer: the single source of truth for what the analyst is allowed to talk
about. The LLM never invents metric definitions - it picks from this catalog, and the
compiler turns those picks into SQL. This is what keeps "success rate" meaning the
same thing on every answer.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Logical tables exposed to the analyst -> physical source. User/LLM SQL may only
# reference the logical names; the guard rewrites them (and applies row-level scope).
LOGICAL_TABLES = {
    "transactions": "raw.transactions_clean",   # de-duplicated (default)
    "transactions_raw": "raw.transactions",     # as ingested, may contain duplicates
    "merchants": "raw.merchants",
    "banks": "raw.banks",
    "payment_methods": "raw.payment_methods",
    "settlements": "raw.settlements",
}
FACT_TABLES = {"transactions", "transactions_raw"}

TABLE_DOCS = {
    "transactions": "One row per payment attempt (de-duplicated on txn_id). Columns: txn_id, "
                    "created_at TIMESTAMP, merchant_id, bank_id (NULL for wallet), method_id "
                    "(UPI|CARD|NB|WALLET), amount_inr, status (SUCCESS|FAILED|PENDING), "
                    "failure_reason, customer_vpa (PII), ingested_at",
    "transactions_raw": "Same as transactions but exactly as ingested; may contain duplicate txn_id rows.",
    "merchants": "merchant_id, merchant_name, category, city_tier, onboarded_on, kam_portfolio",
    "banks": "bank_id, bank_name, bank_tier",
    "payment_methods": "method_id, method_name, is_bank_routed",
    "settlements": "Finance system daily rollup: settlement_date, merchant_id, attempted_count, "
                   "success_count, settled_amount_inr. Resolves PENDING later, so counts differ.",
}

PII_COLUMNS = {"customer_vpa"}


@dataclass(frozen=True)
class Metric:
    key: str
    label: str
    sql: str              # aggregate expression over alias t
    fmt: str = "pct"      # pct | int | inr
    is_rate: bool = False
    numerator: str | None = None
    denominator: str | None = None
    definition: str = ""


METRICS: dict[str, Metric] = {
    "success_rate": Metric(
        "success_rate", "Success rate",
        "AVG(CASE WHEN t.status = 'SUCCESS' THEN 1.0 ELSE 0.0 END)",
        "pct", True,
        "SUM(CASE WHEN t.status = 'SUCCESS' THEN 1 ELSE 0 END)", "COUNT(*)",
        "successful attempts / all attempts (PENDING counted in denominator)"),
    "success_rate_excl_pending": Metric(
        "success_rate_excl_pending", "Success rate (excl. pending)",
        "SUM(CASE WHEN t.status = 'SUCCESS' THEN 1.0 ELSE 0.0 END) / "
        "NULLIF(SUM(CASE WHEN t.status <> 'PENDING' THEN 1 ELSE 0 END), 0)",
        "pct", True,
        "SUM(CASE WHEN t.status = 'SUCCESS' THEN 1 ELSE 0 END)",
        "SUM(CASE WHEN t.status <> 'PENDING' THEN 1 ELSE 0 END)",
        "successful attempts / terminal attempts (SUCCESS + FAILED)"),
    "failure_rate": Metric(
        "failure_rate", "Failure rate",
        "AVG(CASE WHEN t.status = 'FAILED' THEN 1.0 ELSE 0.0 END)", "pct", True,
        "SUM(CASE WHEN t.status = 'FAILED' THEN 1 ELSE 0 END)", "COUNT(*)",
        "failed attempts / all attempts"),
    "attempts": Metric("attempts", "Attempts", "COUNT(*)", "int",
                       definition="count of payment attempts"),
    "successes": Metric("successes", "Successful txns",
                        "SUM(CASE WHEN t.status = 'SUCCESS' THEN 1 ELSE 0 END)", "int",
                        definition="count of SUCCESS attempts"),
    "gmv": Metric("gmv", "GMV (INR)",
                  "SUM(CASE WHEN t.status = 'SUCCESS' THEN t.amount_inr ELSE 0 END)", "inr",
                  definition="sum of amount_inr over SUCCESS attempts"),
    "avg_ticket": Metric("avg_ticket", "Avg ticket (INR)",
                         "AVG(CASE WHEN t.status = 'SUCCESS' THEN t.amount_inr END)", "inr",
                         definition="mean amount_inr of SUCCESS attempts"),
}


@dataclass(frozen=True)
class Dimension:
    key: str
    label: str
    sql: str
    joins: tuple[str, ...] = field(default_factory=tuple)


_BANK_LABEL = ("COALESCE(b.bank_name, CASE WHEN t.method_id = 'WALLET' THEN 'N/A (wallet)' "
               "ELSE 'Unknown (missing bank_id)' END)")

DIMENSIONS: dict[str, Dimension] = {
    "bank": Dimension("bank", "Bank", _BANK_LABEL, ("banks",)),
    "bank_tier": Dimension("bank_tier", "Bank tier", "COALESCE(b.bank_tier, 'n/a')", ("banks",)),
    "method": Dimension("method", "Payment method", "pm.method_name", ("payment_methods",)),
    "merchant": Dimension("merchant", "Merchant", "m.merchant_name", ("merchants",)),
    "category": Dimension("category", "Merchant category", "m.category", ("merchants",)),
    "city_tier": Dimension("city_tier", "City tier", "m.city_tier", ("merchants",)),
    "failure_reason": Dimension("failure_reason", "Failure reason",
                                "COALESCE(t.failure_reason, '(not failed)')"),
    "day": Dimension("day", "Day", "CAST(t.created_at AS DATE)"),
    "week": Dimension("week", "Week starting", "CAST(date_trunc('week', t.created_at) AS DATE)"),
    "hour": Dimension("hour", "Hour of day", "EXTRACT(hour FROM t.created_at)"),
    # PII - exists so the question can be planned; the guard decides who may run it
    "customer_vpa": Dimension("customer_vpa", "Customer VPA", "t.customer_vpa"),
}

JOIN_SQL = {
    "banks": "LEFT JOIN banks b ON t.bank_id = b.bank_id",
    "payment_methods": "LEFT JOIN payment_methods pm ON t.method_id = pm.method_id",
    "merchants": "LEFT JOIN merchants m ON t.merchant_id = m.merchant_id",
}

# filters the planner may use: dimension -> sql column used in WHERE
FILTERABLE = {
    "bank": "b.bank_name", "method": "pm.method_name", "merchant": "m.merchant_name",
    "category": "m.category", "city_tier": "m.city_tier", "bank_tier": "b.bank_tier",
}

# Dimensions worth scanning when explaining a change
DRIVER_DIMENSIONS = ["method", "bank", "category", "merchant", "city_tier"]


def catalog_text() -> str:
    """Compact catalog handed to the LLM so it can only choose known things."""
    lines = ["METRICS:"]
    for m in METRICS.values():
        lines.append(f"  {m.key}: {m.label} - {m.definition}")
    lines.append("DIMENSIONS: " + ", ".join(DIMENSIONS))
    lines.append("FILTERABLE DIMENSIONS: " + ", ".join(FILTERABLE))
    lines.append("TABLES (for free-form SQL only):")
    for k, v in TABLE_DOCS.items():
        lines.append(f"  {k}: {v}")
    return "\n".join(lines)
