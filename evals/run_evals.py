"""
Known-answer evaluation.

Ground truth is computed with independent hand-written SQL directly on raw tables (NOT the
engine's compiler), then compared with what the engine answers. Also checks behaviour:
clarification, access denials, guard blocks and planted-driver detection.

    python evals/run_evals.py            # deterministic planner
    LLM_PROVIDER=gemini GEMINI_API_KEY=... python evals/run_evals.py --planner llm
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import duckdb  # noqa: E402

from analyst.engine import AnalystEngine  # noqa: E402

DB = os.path.join(os.path.dirname(__file__), "..", "data", "payments.duckdb")


def truth(con, sql):
    return con.execute(sql).fetchone()[0]


def pcts(text):
    return [float(x) / 100 for x in re.findall(r"(\d+\.\d+)%", text)]


def main(planner: str, paraphrases: bool = False):
    if planner in ("llm", "auto"):
        # free-tier keys allow ~15 requests/min; pace calls so the run measures the model, not the quota
        os.environ.setdefault("LLM_MIN_INTERVAL_S", "4.5")
    eng = AnalystEngine(DB)
    con = duckdb.connect(DB, read_only=True)
    clean = "raw.transactions_clean"
    def sr(tbl, a, b, extra=""):
        return truth(con, f"""
        SELECT AVG((t.status='SUCCESS')::INT) FROM {tbl} t {extra.split('|')[0] if extra else ''}
        WHERE t.created_at >= DATE '{a}' AND t.created_at < DATE '{b}' + 1 {extra.split('|')[1] if extra else ''}""")

    T_LW = sr(clean, "2026-10-19", "2026-10-25")
    T_PW = sr(clean, "2026-10-12", "2026-10-18")
    T_LW_RAW = sr("raw.transactions", "2026-10-19", "2026-10-25")
    T_UPI_OCT = sr(clean, "2026-10-01", "2026-10-25", "|AND t.method_id='UPI'")
    T_CARD_OCT = sr(clean, "2026-10-01", "2026-10-25", "|AND t.method_id='CARD'")
    T_SEP = sr(clean, "2026-09-01", "2026-09-30")
    T_DELTA_LW = sr(clean, "2026-10-19", "2026-10-25", "|AND t.bank_id='B04'")

    T_SHARE_5K = truth(con, f"""SELECT AVG((amount_inr > 5000)::INT) FROM {clean}
        WHERE created_at >= DATE '2026-10-19' AND created_at < DATE '2026-10-26'""")
    T_UPI_FAIL_2K = truth(con, f"""SELECT COUNT(*) FROM {clean} WHERE method_id = 'UPI' AND status = 'FAILED'
        AND amount_inr > 2000 AND created_at >= DATE '2026-10-01' AND created_at < DATE '2026-11-01'""")

    T_WEEKEND_LW = sr(clean, "2026-10-24", "2026-10-25")
    T_UPI_LW = sr(clean, "2026-10-19", "2026-10-25", "|AND t.method_id='UPI'")
    T_UPI_PW = sr(clean, "2026-10-12", "2026-10-18", "|AND t.method_id='UPI'")

    def share(a, b, cond):
        return truth(con, f"""SELECT AVG(({cond})::INT) FROM {clean}
            WHERE created_at >= DATE '{a}' AND created_at < DATE '{b}' + 1""")
    tech = "status = 'FAILED' AND failure_reason IN ('TIMEOUT', 'BANK_DOWNTIME')"
    bus = "status = 'FAILED' AND failure_reason NOT IN ('TIMEOUT', 'BANK_DOWNTIME')"
    T_TECH_PTS = -(share("2026-10-19", "2026-10-25", tech) - share("2026-10-12", "2026-10-18", tech)) * 100
    T_BUS_PTS = -(share("2026-10-19", "2026-10-25", bus) - share("2026-10-12", "2026-10-18", bus)) * 100
    T_TSR_DELTA_LW = truth(con, f"""SELECT SUM((status = 'SUCCESS')::INT) * 1.0 / SUM(CASE WHEN {bus} THEN 0 ELSE 1 END)
        FROM {clean} WHERE bank_id = 'B04' AND created_at >= DATE '2026-10-19' AND created_at < DATE '2026-10-26'""")

    def result_has(a, target, tol):
        """Any numeric cell in the first row of a free-SQL result equals target (shares also as %)."""
        df = a.tables.get("Result")
        if df is None or not len(df):
            return False
        vals = [float(v) for v in df.iloc[0].tolist() if isinstance(v, (int, float)) and not isinstance(v, bool)]
        return any(abs(v - target) <= tol or abs(v / 100 - target) <= tol for v in vals)

    def custom_condition(a, target, tol, needle):
        # without an LLM the engine must refuse to drop the condition; with one, the SQL must be right
        if planner == "rules":
            return a.status == "clarify" and needle in a.message
        return a.status == "answered" and (a.plan or {}).get("intent") == "free_sql" and result_has(a, target, tol)

    def close(xs, target, tol=0.0002):
        return any(abs(x - target) <= tol for x in xs)

    cases = [
        # (id, question, role, check(answer) -> bool, description)
        ("headline_wow", "Why did success rate drop last week?", "head_of_payments",
         lambda a: a.status == "answered" and close(pcts(a.headline), T_LW) and close(pcts(a.headline), T_PW),
         f"headline matches ground truth {T_PW:.2%} -> {T_LW:.2%}"),
        ("driver_mix", "Why did success rate drop last week?", "head_of_payments",
         lambda a: any("Delta Co-op Bank · UPI" in f["text"] and "Mix effect" in f["text"] for f in a.facts),
         "identifies Delta Co-op UPI mix shift (planted)"),
        ("driver_surge", "Why did success rate drop last week?", "head_of_payments",
         lambda a: any("QuickTix" in f["text"] for f in a.facts), "identifies QuickTix volume surge (planted)"),
        ("driver_rate", "Why did success rate drop last week?", "head_of_payments",
         lambda a: any("Coastal Bank · NetBanking" in f["text"] and "Genuine" in f["text"] for f in a.facts),
         "identifies Coastal NB genuine degradation (planted)"),
        ("no_false_rate_driver", "Why did success rate drop last week?", "head_of_payments",
         lambda a: not any(f["text"].startswith("Genuine performance change in UPI") for f in a.facts),
         "does not mislabel the mix shift as a UPI rate drop"),
        ("dq_dups_caveat", "Why did success rate drop last week?", "head_of_payments",
         lambda a: any("duplicate" in c for c in a.caveats), "duplicate records surfaced as caveat"),
        ("dq_gap_caveat", "Why did success rate drop last week?", "head_of_payments",
         lambda a: any("14 Oct" in c for c in a.caveats), "missing-data day surfaced as caveat"),
        ("raw_vs_clean", "Success rate last week raw including duplicates", "head_of_payments",
         lambda a: close(pcts(a.headline), T_LW_RAW), f"raw success rate {T_LW_RAW:.2%}"),
        ("breakdown_method", "Compare UPI vs Card success rate in October", "head_of_payments",
         lambda a: all(close(pcts(" ".join(f["text"] for f in a.facts)), t) for t in (T_CARD_OCT, T_UPI_OCT))
         or (abs(a.tables[next(iter(a.tables))].set_index("method").success_rate["UPI"] - T_UPI_OCT) < 2e-4),
         f"UPI {T_UPI_OCT:.2%} / Card {T_CARD_OCT:.2%} in October"),
        ("single_number", "What was the success rate in September?", "head_of_payments",
         lambda a: close(pcts(a.headline), T_SEP), f"September success rate {T_SEP:.2%}"),
        ("bank_breakdown", "Success rate by bank last week", "head_of_payments",
         lambda a: abs(a.tables[next(iter(a.tables))].set_index("bank").success_rate["Delta Co-op Bank"]
                       - T_DELTA_LW) < 2e-4, f"Delta Co-op last week {T_DELTA_LW:.2%}"),
        ("ambiguous_time", "Why did success rate drop?", "head_of_payments",
         lambda a: a.status == "clarify", "asks which period"),
        ("ambiguous_metric", "How are we performing last week?", "head_of_payments",
         lambda a: a.status == "clarify", "asks which metric/denominator"),
        ("out_of_range", "Why did success rate drop in March 2025?", "head_of_payments",
         lambda a: a.status == "clarify" and "No data" in a.message, "refuses period outside data"),
        ("pii_denied", "Show customer VPAs with the most failures last week", "head_of_payments",
         lambda a: a.status == "denied" and a.denial_rule == "pii", "PII blocked for non-risk role"),
        ("pii_allowed", "Show customer VPAs with the most failures last week", "risk_ops",
         lambda a: a.status == "answered", "PII allowed for risk ops"),
        ("row_scope_denied", "Success rate for QuickTix last week", "kam_portfolio_a",
         lambda a: a.status == "denied" and a.denial_rule == "row_scope", "merchant outside portfolio blocked"),
        ("row_scope_applied", "Success rate by merchant last week", "kam_portfolio_a",
         lambda a: a.status == "answered" and "QuickTix" not in set(a.tables[next(iter(a.tables))].merchant),
         "portfolio-scoped results only"),
        ("sources_disagree", "Do settlements agree with transactions on success rate last week?",
         "head_of_payments", lambda a: "disagree" in a.headline and any("PENDING" in h for h in a.hypotheses),
         "explains source disagreement"),
        ("custom_sql_share", "What share of transactions were above 5000 INR last week?", "head_of_payments",
         lambda a: custom_condition(a, T_SHARE_5K, 2e-4, "5000"),
         f"amount threshold applied, not dropped (share {T_SHARE_5K:.2%})"),
        ("custom_sql_count", "How many failed UPI transactions over 2000 INR were there in October?",
         "head_of_payments", lambda a: custom_condition(a, T_UPI_FAIL_2K, 0.5, "2000"),
         f"amount threshold applied, not dropped (count {T_UPI_FAIL_2K:,})"),
        ("premise_check", "Why did GMV drop last week?", "head_of_payments",
         lambda a: a.status == "answered" and "actually rose" in a.headline,
         "corrects a false premise (GMV rose)"),
        ("action_question", "What action should we take on success rate last week?", "head_of_payments",
         lambda a: a.status == "answered" and len(a.recommendations) >= 2, "returns recommendations"),
        ("word_cond_weekend", "What was the success rate on weekends last week?", "head_of_payments",
         lambda a: custom_condition(a, T_WEEKEND_LW, 2e-4, "weekends"),
         f"weekend condition applied inside last week, 24-25 Oct ({T_WEEKEND_LW:.2%})"),
        ("word_cond_vague", "Success rate for high-value payments last week", "head_of_payments",
         lambda a: (a.status == "clarify" and "high-value" in a.message) if planner == "rules" else
         (a.status == "answered" and (a.plan or {}).get("intent") == "free_sql"
          and any("no standard definition" in c for c in a.caveats)),
         "vague threshold never silently ignored; definition disclosed"),
        ("outcome_split", "Why did success rate drop last week?", "head_of_payments",
         lambda a: any(f"technical failures {T_TECH_PTS:+.2f} pts" in f["text"]
                       and f"business declines {T_BUS_PTS:+.2f} pts" in f["text"] for f in a.facts)
         and any("mainly technical" in h for h in a.hypotheses),
         f"splits the drop: technical {T_TECH_PTS:+.2f} pts vs business {T_BUS_PTS:+.2f} pts"),
        ("technical_sr", "Technical success rate by bank last week", "head_of_payments",
         lambda a: a.status == "answered" and abs(a.tables[next(iter(a.tables))].set_index("bank")
                                                   .technical_success_rate["Delta Co-op Bank"] - T_TSR_DELTA_LW) < 2e-4,
         f"technical success rate excludes business declines (Delta Co-op {T_TSR_DELTA_LW:.2%})"),
        ("unknown_term", "Success rate for premium customers last week", "head_of_payments",
         lambda a: (a.status == "clarify" and "premium" in a.message) if planner == "rules" else
         (a.status == "clarify" or (a.plan or {}).get("intent") == "free_sql"
          or "premium" in json.dumps(a.plan or {}).lower() or a.confidence.get("level") == "low"),
         "an unrecognised word is never silently ignored"),
        ("hour_labels", "Success rate by hour of day last week", "head_of_payments",
         lambda a: a.status == "answered" and str(a.tables[next(iter(a.tables))].iloc[0, 0]) == "00:00",
         "hours shown as HH:00"),
    ]
    sql_cases = [
        ("guard_delete", "DELETE FROM transactions", "blocked"),
        ("guard_multi", "SELECT 1; DROP TABLE banks", "blocked"),
        ("guard_file", "SELECT * FROM read_csv_auto('/etc/passwd')", "blocked"),
        ("guard_schema", "SELECT COUNT(*) FROM raw.transactions", "blocked"),
        ("guard_unlisted_fn", "SELECT current_setting('threads')", "blocked"),
        ("guard_cartesian", "SELECT COUNT(*) FROM transactions a, transactions b", "blocked"),
        ("guard_star_pii", "SELECT * FROM transactions", "denied"),
        ("guard_ok", "SELECT COUNT(*) AS n FROM transactions", "answered"),
    ]

    passed = 0
    provider_errs = 0
    total = len(cases) + len(sql_cases) + 1
    cache = {}
    for cid, q, role, check, desc in cases:
        a = cache.get((q, role)) or eng.ask(q, role, planner)
        cache[(q, role)] = a
        try:
            ok = bool(check(a))
        except Exception as e:  # noqa: BLE001
            ok = False
            desc += f" [check error: {e}]"
        passed += ok
        provider_err = a.status == "error" and "provider unavailable" in a.message
        provider_errs += (not ok) and provider_err
        tag = "PASS" if ok else ("PROVIDER" if provider_err else "FAIL")
        print(f"{tag:<8}  {cid:<22} {desc}" + ("  [model provider error, not a wrong answer]" if tag == "PROVIDER" else ""))
    followups = [
        ("followup_breakdown", "Why did success rate drop last week?", "Now break that down by bank",
         lambda a: a.status == "answered" and abs(a.tables[next(iter(a.tables))].set_index("bank")
                                                   .success_rate["Delta Co-op Bank"] - T_DELTA_LW) < 2e-4,
         f"keeps metric + period from the previous question (Delta Co-op {T_DELTA_LW:.2%})"),
        ("followup_slice", "Why did success rate drop last week?", "What about UPI?",
         lambda a: a.status == "answered" and close(pcts(a.headline), T_UPI_LW) and close(pcts(a.headline), T_UPI_PW),
         f"re-runs the diagnosis on UPI only ({T_UPI_PW:.2%} -> {T_UPI_LW:.2%})"),
    ]
    total += len(followups)
    for cid, q1, q2, check, desc in followups:
        first = cache.get((q1, "head_of_payments")) or eng.ask(q1, "head_of_payments", planner)
        a = eng.ask(q2, "head_of_payments", planner, previous={"question": q1, "plan": first.plan})
        try:
            ok = bool(check(a))
        except Exception as e:  # noqa: BLE001
            ok = False
            desc += f" [check error: {e}]"
        passed += ok
        provider_err = a.status == "error" and "provider unavailable" in a.message
        provider_errs += (not ok) and provider_err
        tag = "PASS" if ok else ("PROVIDER" if provider_err else "FAIL")
        print(f"{tag:<8}  {cid:<22} {desc}")
    for cid, sql, expect in sql_cases:
        a = eng.run_sql(sql, "head_of_payments")
        ok = a.status == expect
        passed += ok
        print(f"{'PASS' if ok else 'FAIL'}  {cid:<22} expect {expect}, got {a.status} {a.denial_rule or ''}")

    # provider outage mid-workflow -> graceful fallback
    provider_before = eng.llm.provider
    eng.llm.provider, eng.llm.simulate_outage = "gemini", True
    a = eng.ask("Why did success rate drop last week?", "head_of_payments", "auto")
    eng.llm.provider, eng.llm.simulate_outage = provider_before, False
    ok = a.status == "answered" and any(s.status == "fallback" for s in a.steps)
    passed += ok
    print(f"{'PASS' if ok else 'FAIL'}  {'provider_outage':<22} falls back to deterministic planner")

    print(f"\n{passed}/{total} passed" + (f" ({provider_errs} not passed due to model-provider errors)"
                                         if provider_errs else ""))
    if paraphrases:
        run_paraphrases(eng, planner)
    return 0 if passed == total else 1


# Held-out rewordings: never used to write the planner prompt. They check the plan shape
# (intent, metric, dimensions, window), so they measure how well the planner generalises.
PARAPHRASES = [
    ("What caused the dip in payment success over the past week?", "answered",
     dict(intent="diagnose_change", metric="success_rate", window=("2026-10-19", "2026-10-25"))),
    ("How did Card do compared with NetBanking in September?", "answered",
     dict(intent="breakdown", dims=["method"], window=("2026-09-01", "2026-09-30"))),
    ("Give me the daily success rate trend for October", "answered",
     dict(intent="trend", window=("2026-10-01", "2026-10-25"))),
    ("GMV by category last week", "answered", dict(intent="breakdown", metric="gmv", dims=["category"])),
    ("Is the finance team's settlement data consistent with our transaction logs for last week?", "answered",
     dict(intent="compare_sources")),
    ("Why is success rate down?", "clarify", dict(intent="clarify")),
    ("Success rate excluding pending for Coastal Bank last week", "answered",
     dict(intent="breakdown", metric="success_rate_excl_pending")),
    ("Why did success rate fall in January 2024?", "clarify", dict()),
    ("Which merchants had the highest failure rate last week?", "answered",
     dict(intent="breakdown", metric="failure_rate", dims=["merchant"])),
    ("Top failure reasons last week", "answered", dict(intent="breakdown", dims=["failure_reason"])),
]


def run_paraphrases(eng, planner):
    print("\nHeld-out paraphrases (plan shape):")
    ok_n = 0
    for q, status, exp in PARAPHRASES:
        a = eng.ask(q, "head_of_payments", planner)
        p = a.plan or {}
        why = []
        if a.status != status:
            why.append(f"status {a.status}" + (f" ({a.message[:70]})" if a.status == "error" else ""))
        if "intent" in exp and p.get("intent") != exp["intent"]:
            why.append(f"intent {p.get('intent')}")
        if "metric" in exp and p.get("metric") != exp["metric"]:
            why.append(f"metric {p.get('metric')}")
        if "dims" in exp and list(p.get("dimensions") or []) != exp["dims"]:
            why.append(f"dims {p.get('dimensions')}")
        if "window" in exp:
            w = p.get("window") or {}
            if (w.get("start"), w.get("end")) != exp["window"]:
                why.append(f"window {w.get('start')}..{w.get('end')}")
        ok_n += not why
        print(f"{'PASS' if not why else 'FAIL'}  {q[:60]:<60} {'; '.join(why)}")
    print(f"{ok_n}/{len(PARAPHRASES)} paraphrases planned correctly")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--planner", default="rules", choices=["rules", "llm", "auto"])
    ap.add_argument("--paraphrases", action="store_true", help="also run held-out rewordings")
    args = ap.parse_args()
    sys.exit(main(args.planner, args.paraphrases))
