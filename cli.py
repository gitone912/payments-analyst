"""Quick terminal client: python cli.py "why did success rate drop last week?" [--role kam_portfolio_a]"""
import argparse
import os

from analyst.engine import AnalystEngine

DB = os.path.join(os.path.dirname(__file__), "data", "payments.duckdb")


def show(a):
    print(f"\n=== [{a.status}] {a.question}  (role: {a.role}, planner: {a.planner_used}, {a.total_ms:.0f} ms)")
    if a.message:
        print("MESSAGE:", a.message)
    if a.clarification:
        print("OPTIONS:", a.clarification.get("options"))
    for s in a.steps:
        print(f"  step {s.name:<13} {s.status:<8} {s.detail}")
    for f in a.facts:
        print("  FACT  ", f["text"], f["evidence"])
    for h in a.hypotheses:
        print("  HYPO  ", h)
    for r in a.recommendations:
        print("  REC   ", r)
    for c in a.caveats:
        print("  CAVEAT", c)
    for x in a.assumptions:
        print("  ASSUME", x)
    if a.confidence:
        print("  CONFIDENCE", a.confidence)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="+")
    ap.add_argument("--role", default="head_of_payments")
    ap.add_argument("--planner", default="auto")
    ap.add_argument("--sql", action="store_true", help="print executed SQL")
    ap.add_argument("--then", action="append", default=[], metavar="FOLLOW_UP",
                    help='follow-up question(s), e.g. --then "now break that down by bank"')
    a = ap.parse_args()
    eng = AnalystEngine(DB)
    previous = None
    for question in [" ".join(a.question), *a.then]:
        print(f"\n>>> {question}")
        for ans in eng.ask_parts(question, a.role, a.planner, previous=previous):
            show(ans)
            if a.sql:
                for q in ans.queries:
                    print(f"\n-- {q.query_id} {q.purpose} ({q.ms:.0f} ms, {q.rows} rows)\n{q.sql}")
            if ans.status == "answered":
                previous = {"question": ans.question, "plan": ans.plan}
