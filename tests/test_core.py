"""Unit tests: decomposition maths, LLM retry/validation on malformed output, provider outage."""
import os
import sys
from datetime import date

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from analyst.drivers import decompose  # noqa: E402
from analyst.llm import LLMPlanner, ProviderUnavailable  # noqa: E402


def test_decomposition_sums_exactly():
    df = pd.DataFrame([
        ("A", "baseline", 900, 810), ("B", "baseline", 100, 70),
        ("A", "current", 900, 810), ("B", "current", 600, 420),     # pure mix: rates unchanged
        ("C", "current", 50, 25),                                     # new segment
    ], columns=["segment", "period", "denom", "numer"])
    d = decompose(df, "x")
    assert abs(d.mix_total + d.rate_total - d.delta) < 1e-12
    assert abs(d.segments.set_index("segment").rate_effect[["A", "B"]]).max() < 1e-12


def _planner():
    p = LLMPlanner(date(2026, 8, 31), date(2026, 10, 25), ["Coastal Bank"], ["ecommerce"], provider="gemini")
    return p


def test_malformed_then_valid_output_retries():
    p = _planner()
    outputs = iter([
        "Sure! Here is your plan: intent = diagnose",                 # not JSON
        '{"intent":"diagnose_change","metric":"success_rate","dimensions":[],"filters":{},'
        '"window":{"start":"2026-10-19","end":"2026-10-25"},'
        '"baseline":{"start":"2026-10-12","end":"2026-10-18"}}',
    ])
    p._complete = lambda user: (next(outputs), 100, 50)
    res = p.plan("why did sr drop last week")
    assert res.plan is not None and res.plan.intent == "diagnose_change"
    assert len(res.calls) == 2 and not res.calls[0].ok and res.calls[1].ok


def test_unsupported_metric_rejected_then_gives_up():
    p = _planner()
    bad = ('{"intent":"breakdown","metric":"customer_happiness","dimensions":["bank"],"filters":{},'
           '"window":{"start":"2026-10-19","end":"2026-10-25"}}')
    p._complete = lambda user: (bad, 100, 50)
    res = p.plan("happiness by bank")
    assert res.plan is None and len(res.calls) == p.max_attempts
    assert "unknown metric" in res.errors[0]


def test_provider_outage_returns_without_retry_storm():
    p = _planner()

    def boom(user):
        raise ProviderUnavailable("HTTP 503 from provider")
    p._complete = boom
    res = p.plan("anything")
    assert res.plan is None and len(res.calls) == 1


def test_trailing_comma_json_is_repaired():
    from analyst.llm import plan_from_json
    p = plan_from_json('{"intent": "clarify", "clarification": {"question": "q", "options": ["a",],},}')
    assert p.intent == "clarify" and p.clarification["options"] == ["a"]


def test_successful_plan_is_cached():
    p = _planner()
    calls = []
    good = ('{"intent":"breakdown","metric":"success_rate","dimensions":[],"filters":{},'
            '"window":{"start":"2026-10-19","end":"2026-10-25"},"baseline":null}')
    p._complete = lambda user: (calls.append(user), (good, 10, 5))[1]
    assert p.plan("Success rate last week").plan is not None
    r = p.plan("  success RATE last week ")
    assert r.plan is not None and len(calls) == 1 and r.calls[0].attempt == 0


def test_ambiguity_guard_overrides_llm_guess():
    """'How are we performing' names no measure: even if the LLM picks one, the engine must clarify."""
    from analyst.engine import Answer, AnalystEngine
    from analyst.llm import LLMResult
    from analyst.plan import Plan, Window
    db = os.path.join(os.path.dirname(__file__), "..", "data", "payments.duckdb")
    if not os.path.exists(db):
        import pytest
        pytest.skip("run data/generate_data.py first")
    eng = AnalystEngine(db)
    eng.llm.provider = "gemini"
    guess = Plan(intent="breakdown", metric="success_rate",
                 window=Window(date(2026, 10, 19), date(2026, 10, 25)))
    eng.llm.plan = lambda q, **kw: LLMResult(plan=guess)
    plan = eng.make_plan("How are we performing last week?", Answer("q", "r"), "llm")
    assert plan.intent == "clarify"


def test_unmodelled_conditions_detector():
    from analyst.rule_planner import unmodelled_conditions as u
    assert u("What share of transactions are above 5000 INR last week?")
    assert u("failed UPI txns over ₹2,000") and u("amount > 10k") and u("median ticket size")
    # time phrases are not conditions the catalog misses
    for q in ("Success rate over the past week", "last 14 days", "more than 2 weeks ago", "Success rate by bank"):
        assert not u(q), q


def test_premise_direction():
    from analyst.rule_planner import premise_direction as d
    assert d("Why did GMV drop last week?") == "down"
    assert d("Why did failure rate go up?") == "up"
    assert d("What drove the change?") is None


def _engine():
    from analyst.engine import AnalystEngine
    db = os.path.join(os.path.dirname(__file__), "..", "data", "payments.duckdb")
    if not os.path.exists(db):
        import pytest
        pytest.skip("run data/generate_data.py first")
    return AnalystEngine(db)


def test_word_conditions_detected():
    from analyst.rule_planner import unmodelled_conditions as u
    for q in ("Success rate for high-value payments last week", "Success rate on weekends last week",
              "Success rate between 9am and 5pm last week", "Success rate for first-time customers"):
        assert u(q), q
    for q in ("Success rate by hour of day last week", "Daily success rate trend for October"):
        assert not u(q), q


def test_period_guard_rejects_wrong_llm_window():
    """'weekends last week' answered with 18-19 Oct (Sun+Mon, outside last week) must not get through."""
    from analyst.engine import Answer
    from analyst.llm import LLMResult
    from analyst.plan import Plan
    eng = _engine()
    eng.llm.provider = "gemini"
    wrong = Plan(intent="free_sql", sql="SELECT COUNT(*) FROM transactions WHERE created_at >= '2026-10-18' "
                                        "AND created_at < '2026-10-20' AND dayofweek(created_at) IN (0, 6)")
    right = Plan(intent="free_sql", sql="SELECT COUNT(*) FROM transactions WHERE created_at >= '2026-10-24' "
                                        "AND created_at < '2026-10-26'")
    calls = []

    def fake(q, feedback=None, context=None):
        calls.append(feedback)
        return LLMResult(plan=right if feedback and "period check" in feedback else wrong)
    eng.llm.plan = fake
    ans = Answer("q", "r")
    ans.planner_used = "LLM"
    plan = eng._period_guard("Success rate on weekends last week", wrong, ans, "llm")
    assert plan.sql == right.sql and any(s.name == "period_check" and s.status == "fixed" for s in ans.steps)

    eng.llm.plan = lambda q, **kw: LLMResult(plan=wrong)  # model never fixes it -> deterministic fallback
    ans = Answer("q", "r")
    plan = eng._period_guard("Success rate on weekends last week", wrong, ans, "llm")
    assert plan.intent == "clarify"  # rules can't do weekends, so they ask rather than guess


def test_followup_inherits_previous_plan():
    eng = _engine()
    first = eng.ask("Why did success rate drop last week?", "head_of_payments", "rules")
    prev = {"question": first.question, "plan": first.plan}
    a = eng.ask("Now break that down by bank", "head_of_payments", "rules", previous=prev)
    assert a.status == "answered" and a.plan["dimensions"] == ["bank"]
    assert a.plan["window"] == first.plan["window"] and a.plan["metric"] == "success_rate"
    b = eng.ask("What about UPI?", "head_of_payments", "rules", previous=prev)
    assert b.plan["intent"] == "diagnose_change" and b.plan["filters"] == {"method": ["UPI"]}
    # a standalone question ignores the previous plan
    c = eng.ask("Success rate by method in September", "head_of_payments", "rules", previous=prev)
    assert c.plan["window"]["start"] == "2026-09-01"


def test_vague_threshold_definition_is_surfaced():
    from analyst.engine import Answer
    from analyst.plan import Plan
    from analyst.rbac import ROLES
    eng = _engine()
    plan = Plan(intent="free_sql", assumptions=["high-value = amount_inr > 10000"],
                sql="SELECT AVG((status = 'SUCCESS')::FLOAT) AS success_rate FROM transactions "
                    "WHERE amount_inr > 10000 AND created_at >= '2026-10-19' AND created_at < '2026-10-26'")
    ans = Answer("Success rate for high-value payments last week", "r")
    eng._free_sql(plan, ROLES["head_of_payments"], ans)
    assert "amount_inr > 10000" in ans.headline and "%" in ans.headline  # formatted, not 0.927...
    assert any("no standard definition" in c for c in ans.caveats)


def test_amount_is_not_read_as_year():
    from analyst.rule_planner import RulePlanner
    r = RulePlanner(date(2026, 8, 31), date(2026, 10, 25), [], [], [])
    w, _, _ = r.parse_time(" how many failed upi transactions over 2000 inr in october ")
    assert (w.start, w.end) == (date(2026, 10, 1), date(2026, 10, 25))
    w, _, _ = r.parse_time(" success rate in march 2025 ")
    assert w.start == date(2025, 3, 1)
