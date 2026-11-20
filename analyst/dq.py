"""
Data-quality checks run before analysis. Each finding records the dates it affects so
the engine can attach it as a caveat only when it overlaps the analysis window.
Runs on internal connections (not user SQL); results are cached per dataset.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from .executor import Executor


@dataclass
class DQFinding:
    check: str
    status: str             # pass | warn | fail
    summary: str
    affected_days: list[date] = field(default_factory=list)
    detail: dict = field(default_factory=dict)
    handling: str = ""

    def __post_init__(self):
        self.affected_days = [d.date() if hasattr(d, "date") and callable(d.date) else d
                              for d in self.affected_days]

    def overlaps(self, start: date, end: date) -> bool:
        return any(start <= d <= end for d in self.affected_days)


def run_checks(ex: Executor) -> list[DQFinding]:
    out: list[DQFinding] = []

    # 1. duplicates
    d = ex.raw("""
        SELECT CAST(created_at AS DATE) AS day, COUNT(*) - COUNT(DISTINCT txn_id) AS dup_rows,
               COUNT(*) AS raw_rows
        FROM raw.transactions GROUP BY 1 HAVING COUNT(*) > COUNT(DISTINCT txn_id) ORDER BY 1""")
    if len(d):
        impact = ex.raw(f"""
            SELECT
              (SELECT AVG((status='SUCCESS')::INT) FROM raw.transactions
                 WHERE CAST(created_at AS DATE) IN ({', '.join(f"DATE '{x}'" for x in d.day.astype(str))})) AS raw_sr,
              (SELECT AVG((status='SUCCESS')::INT) FROM raw.transactions_clean
                 WHERE CAST(created_at AS DATE) IN ({', '.join(f"DATE '{x}'" for x in d.day.astype(str))})) AS clean_sr""")
        raw_sr, clean_sr = float(impact.raw_sr[0]), float(impact.clean_sr[0])
        out.append(DQFinding(
            "duplicate_records", "warn",
            f"{int(d.dup_rows.sum())} duplicate txn_id rows on {len(d)} day(s) "
            f"({', '.join(x.strftime('%d %b') for x in d.day)}); duplicates are mostly FAILED rows "
            f"re-ingested later. On those days raw success rate is {raw_sr:.2%} vs {clean_sr:.2%} de-duplicated.",
            list(d.day), {"by_day": d.to_dict("records"), "raw_sr": raw_sr, "clean_sr": clean_sr},
            "Analysis uses the de-duplicated view (first ingested row per txn_id)."))
    else:
        out.append(DQFinding("duplicate_records", "pass", "No duplicate txn_id rows."))

    # 2. volume gaps / partial days
    v = ex.raw("""
        SELECT CAST(created_at AS DATE) AS day, COUNT(*) AS n, MAX(EXTRACT(hour FROM created_at)) AS last_hour
        FROM raw.transactions_clean GROUP BY 1 ORDER BY 1""")
    v["median7"] = v.n.rolling(7, min_periods=3).median().shift(1)
    bad = v[(v.n < 0.6 * v.median7) | (v.last_hour < 22)]
    if len(bad):
        out.append(DQFinding(
            "missing_data", "fail",
            "Incomplete data on " + ", ".join(
                f"{r.day:%d %b} ({int(r.n):,} rows vs ~{int(r.median7):,} typical, last hour {int(r.last_hour)}:00)"
                for r in bad.itertuples()) + ". Looks like an ingestion outage.",
            list(bad.day), {"days": bad[["day", "n", "median7", "last_hour"]].to_dict("records")},
            "Counts and GMV for these days are understated. Rates are kept but carry a time-of-day bias."))
    else:
        out.append(DQFinding("missing_data", "pass", "Daily volumes complete."))

    # 3. missing dimension values (bank_id NULL for bank-routed methods)
    nb = ex.raw("""
        SELECT CAST(created_at AS DATE) AS day, SUM((bank_id IS NULL)::INT) AS nulls, COUNT(*) AS n
        FROM raw.transactions_clean WHERE method_id <> 'WALLET' GROUP BY 1
        HAVING SUM((bank_id IS NULL)::INT) > 0 ORDER BY 1""")
    if len(nb):
        out.append(DQFinding(
            "missing_bank_id", "warn",
            f"{int(nb.nulls.sum()):,} non-wallet transactions have no bank_id on "
            f"{', '.join(x.strftime('%d %b') for x in nb.day)} (up to {(nb.nulls / nb.n).max():.1%} of a day).",
            list(nb.day), {"by_day": nb.to_dict("records")},
            "Shown as 'Unknown (missing bank_id)' in bank breakdowns. Wallet rows have no bank by design."))
    else:
        out.append(DQFinding("missing_bank_id", "pass", "bank_id present for all bank-routed methods."))

    # 4. source reconciliation transactions vs settlements
    r = ex.raw("""
        WITH tx AS (SELECT CAST(created_at AS DATE) AS day,
                           SUM((status='SUCCESS')::INT) AS tx_s, SUM((status='PENDING')::INT) AS tx_p
                    FROM raw.transactions_clean GROUP BY 1),
             st AS (SELECT settlement_date AS day, SUM(success_count) AS st_s FROM raw.settlements GROUP BY 1)
        SELECT tx.day, tx_s, st_s, tx_p, (st_s - tx_s) * 1.0 / tx_s AS rel_diff
        FROM tx JOIN st USING (day) ORDER BY 1""")
    mism = r[r.rel_diff.abs() > 0.005]
    if len(mism):
        explained = float(((r.st_s - r.tx_s) / r.tx_p.clip(lower=1)).median())
        out.append(DQFinding(
            "source_disagreement", "warn",
            f"Settlements report {r.rel_diff.mean():+.2%} more successful transactions than the transactions "
            f"table on average ({len(mism)} of {len(r)} days differ by >0.5%). The gap matches PENDING "
            f"payments that later settled (~{explained:.0%} of pending resolve to success).",
            list(mism.day), {"by_day": r.to_dict("records")},
            "Transactions = real-time authorisation outcome (used for success rate). Settlements = finance "
            "ledger after pending resolution. Use settlements for money questions, transactions for "
            "payment-experience questions."))
    else:
        out.append(DQFinding("source_disagreement", "pass", "Transactions and settlements agree."))

    # 5. pending share
    p = ex.raw("SELECT AVG((status='PENDING')::INT) AS p FROM raw.transactions_clean")
    pend = float(p.p[0])
    out.append(DQFinding(
        "pending_share", "pass" if pend < 0.03 else "warn",
        f"{pend:.1%} of attempts are still PENDING. They count in the success-rate denominator "
        "(use 'success rate excluding pending' for terminal outcomes only).", [], {"pending": pend}))
    return out
