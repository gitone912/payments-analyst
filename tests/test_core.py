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
    p._complete = lambda user, *_: (next(outputs), 100, 50)
    res = p.plan("why did sr drop last week")
    assert res.plan is not None and res.plan.intent == "diagnose_change"
    assert len(res.calls) == 2 and not res.calls[0].ok and res.calls[1].ok


def test_unsupported_metric_rejected_then_gives_up():
    p = _planner()
    bad = ('{"intent":"breakdown","metric":"customer_happiness","dimensions":["bank"],"filters":{},'
           '"window":{"start":"2026-10-19","end":"2026-10-25"}}')
    p._complete = lambda user, *_: (bad, 100, 50)
    res = p.plan("happiness by bank")
    assert res.plan is None and len(res.calls) == p.max_attempts
    assert "unknown metric" in res.errors[0]


def test_provider_outage_returns_without_retry_storm():
    """Every model down: one call per model (no wait hint -> no retry), then give up so the engine
    falls back to the deterministic planner."""
    p = _planner()
    p.models = ["primary", "backup"]
    tried = []

    def boom(user, outage=False, model=None):
        tried.append(model)
        raise ProviderUnavailable("HTTP 503 from provider")
    p._complete = boom
    res = p.plan("anything")
    assert res.plan is None and tried == ["primary", "backup"] and len(res.calls) == 2


GOOD_PLAN = ('{"intent":"breakdown","metric":"success_rate","dimensions":[],"filters":{},'
             '"window":{"start":"2026-10-19","end":"2026-10-25"},"baseline":null}')


def test_backup_model_answers_when_primary_is_overloaded():
    p = _planner()
    p.models = ["primary", "backup"]

    def complete(user, outage=False, model=None):
        if model == "primary":
            raise ProviderUnavailable("HTTP 503 from provider")
        return GOOD_PLAN, 10, 5
    p._complete = complete
    res = p.plan("Success rate last week")
    assert res.plan is not None and res.model == "backup"
    assert [c.model for c in res.calls] == ["primary", "backup"]


def test_short_retry_after_is_honoured_then_same_model_retried():
    p = _planner()
    p.models = ["primary", "backup"]
    slept, calls = [], []
    p._sleep = slept.append

    def complete(user, outage=False, model=None):
        calls.append(model)
        if len(calls) == 1:
            raise ProviderUnavailable("HTTP 429 from provider", retry_after=2.0)
        return GOOD_PLAN, 10, 5
    p._complete = complete
    res = p.plan("Success rate last week")
    assert slept == [2.0] and calls == ["primary", "primary"] and res.model == "primary"


def test_long_retry_after_is_not_waited_for():
    p = _planner()
    p.models = ["primary", "backup"]
    slept = []
    p._sleep = slept.append

    def complete(user, outage=False, model=None):
        if model == "primary":
            raise ProviderUnavailable("HTTP 429 from provider", retry_after=37.0)  # too long for a user
        return GOOD_PLAN, 10, 5
    p._complete = complete
    assert p.plan("Success rate last week").model == "backup" and slept == []


def test_circuit_breaker_skips_a_failing_model_then_recovers():
    from analyst.llm import CircuitBreaker
    now = [0.0]
    p = _planner()
    p.models = ["primary", "backup"]
    p.breaker = CircuitBreaker(threshold=3, cooldown_s=60, clock=lambda: now[0])
    calls = []

    def complete(user, outage=False, model=None):
        calls.append(model)
        if model == "primary":
            raise ProviderUnavailable("HTTP 503 from provider")
        return GOOD_PLAN, 10, 5
    p._complete = complete
    for i in range(3):                       # three failures open the breaker for 'primary'
        p.plan(f"Success rate last week {i}")
    calls.clear()
    p.plan("Success rate last week again")
    assert calls == ["backup"]               # primary skipped while open
    now[0] = 61.0                            # after the cooldown it is tried again
    calls.clear()
    p.plan("Success rate last week later")
    assert calls[0] == "primary"


def test_retry_after_parsed_from_gemini_body():
    from analyst.llm import _retry_after
    assert _retry_after({}, b'{"details":[{"retryDelay":"36s"}]}') == 36.0
    assert _retry_after({}, b"Please retry in 4.5s.") == 4.5
    assert _retry_after({"Retry-After": "7"}, b"") == 7.0
    assert _retry_after({}, b"overloaded") is None


def test_trailing_comma_json_is_repaired():
    from analyst.llm import plan_from_json
    p = plan_from_json('{"intent": "clarify", "clarification": {"question": "q", "options": ["a",],},}')
    assert p.intent == "clarify" and p.clarification["options"] == ["a"]


def test_successful_plan_is_cached():
    p = _planner()
    calls = []
    good = ('{"intent":"breakdown","metric":"success_rate","dimensions":[],"filters":{},'
            '"window":{"start":"2026-10-19","end":"2026-10-25"},"baseline":null}')
    p._complete = lambda user, *_: (calls.append(user), (good, 10, 5))[1]
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
                                        "AND created_at < '2026-10-26' AND dayofweek(created_at) IN (0, 6)")
    calls = []

    def fake(q, feedback=None, context=None, **_):
        calls.append(feedback)
        return LLMResult(plan=right if feedback and "failed validation" in feedback else wrong)
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


def test_concurrent_users_do_not_share_followup_context():
    """One engine serves every app user. Follow-up context must stay with its own request."""
    from concurrent.futures import ThreadPoolExecutor
    eng = _engine()
    a0 = eng.ask("Why did success rate drop last week?", "head_of_payments", "rules")
    b0 = eng.ask("Success rate by method in September", "head_of_payments", "rules")
    prev_a = {"question": a0.question, "plan": a0.plan}
    prev_b = {"question": b0.question, "plan": b0.plan}

    def user_a(_):
        a = eng.ask("What about UPI?", "head_of_payments", "rules", previous=prev_a, user="a")
        return a.plan["window"]["start"] == "2026-10-19" and a.plan["intent"] == "diagnose_change"

    def user_b(_):
        b = eng.ask("What about Card?", "head_of_payments", "rules", previous=prev_b, user="b")
        return b.plan["window"]["start"] == "2026-09-01" and b.plan["intent"] == "breakdown"

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda i: (user_a if i % 2 else user_b)(i), range(24)))
    assert all(results)


def test_audit_log_records_user_and_is_per_user(tmp_path):
    from analyst.engine import AnalystEngine
    db = os.path.join(os.path.dirname(__file__), "..", "data", "payments.duckdb")
    if not os.path.exists(db):
        import pytest
        pytest.skip("run data/generate_data.py first")
    eng = AnalystEngine(db, audit_path=str(tmp_path / "audit.jsonl"))
    eng.ask("Success rate by bank last week", "head_of_payments", "rules", user="alice")
    eng.ask("Success rate by method last week", "risk_ops", "rules", user="bob")
    mine = eng.ex.read_audit(user="alice")
    assert mine and all(r["user"] == "alice" for r in mine)
    assert all(r["user"] == "bob" for r in eng.ex.read_audit(user="bob"))


def test_sql_must_apply_stated_conditions():
    from analyst.plan import Plan
    eng = _engine()
    q = "How many failed UPI transactions over 2000 INR were there in October?"
    dropped = Plan(intent="free_sql", sql="SELECT COUNT(*) FROM transactions WHERE status = 'FAILED' "
                                          "AND created_at >= '2026-10-01' AND created_at < '2026-11-01'")
    applied = Plan(intent="free_sql", sql="SELECT COUNT(*) FROM transactions WHERE status = 'FAILED' AND "
                                          "amount_inr > 2000 AND created_at >= '2026-10-01' AND created_at < '2026-11-01'")
    assert eng._condition_problems(q, dropped) and not eng._condition_problems(q, applied)
    assert eng._condition_problems("Success rate on weekends last week", dropped)


def test_rule_planner_fails_closed_on_unknown_words():
    eng = _engine()
    a = eng.ask("Success rate for premium customers last week", "head_of_payments", "rules")
    assert a.status == "clarify" and "premium" in a.message
    b = eng.ask("Success rate by bank last week", "head_of_payments", "rules")
    assert b.status == "answered"


def test_unaddressed_llm_terms_lower_confidence():
    from analyst.llm import LLMResult
    from analyst.plan import Plan, Window
    eng = _engine()
    eng.llm.provider = "gemini"
    ignores = Plan(intent="breakdown", metric="success_rate", window=Window(date(2026, 10, 19), date(2026, 10, 25)))
    eng.llm.plan = lambda q, **kw: LLMResult(plan=ignores)  # never explains 'festive'
    a = eng.ask("Success rate for festive shoppers last week", "head_of_payments", "llm")
    assert a.status == "answered" and a.confidence["level"] == "low"
    assert any("festive" in c for c in a.caveats)


def test_free_sql_lists_applied_conditions():
    from analyst.engine import Answer
    from analyst.plan import Plan
    from analyst.rbac import ROLES
    eng = _engine()
    ans = Answer("q", "r")
    eng._free_sql(Plan(intent="free_sql", sql="SELECT COUNT(*) AS n FROM transactions WHERE amount_inr > 5000 "
                                                "AND created_at >= '2026-10-19'"), ROLES["head_of_payments"], ans)
    assert any("Conditions applied" in f["text"] and "amount_inr > 5000" in f["text"] for f in ans.facts)


def test_password_auth_and_role_from_account(tmp_path, monkeypatch):
    import json as _json
    from analyst.auth import hash_password, user_from_headers, verify_password
    f = tmp_path / "users.json"
    f.write_text(_json.dumps({"users": [
        {"username": "kam.a", "role": "kam_portfolio_a", **hash_password("pw1")},
        {"username": "rogue", "role": "superuser", **hash_password("pw2")}]}))
    assert verify_password("KAM.A", "pw1", str(f)).role == "kam_portfolio_a"
    assert verify_password("kam.a", "wrong", str(f)) is None
    assert verify_password("nobody", "pw1", str(f)) is None
    assert verify_password("rogue", "pw2", str(f)) is None          # unknown role -> no access
    # header mode: identity from the SSO proxy, role still from the users file (never the request)
    monkeypatch.setenv("AUTH_HEADER", "X-Forwarded-Email")
    assert user_from_headers({"X-Forwarded-Email": "kam.a"}, str(f)).role == "kam_portfolio_a"
    assert user_from_headers({"X-Forwarded-Email": "intruder", "X-Role": "risk_ops"}, str(f)) is None
    assert user_from_headers({}, str(f)) is None


def test_guard_allowlist_blocks_unlisted_functions():
    eng = _engine()
    for sql in ("SELECT current_setting('threads')", "SELECT getenv('HOME')",
                "SELECT COUNT(*) FROM transactions WHERE regexp_matches(failure_reason, '(a+)+$')"):
        a = eng.run_sql(sql, "head_of_payments")
        assert a.status == "blocked" and a.denial_rule == "function_not_allowed", sql
    ok = eng.run_sql("SELECT dayofweek(created_at) AS d, COUNT(*) AS n FROM transactions GROUP BY 1",
                     "head_of_payments")
    assert ok.status == "answered"


def test_outcome_split_is_exact():
    """technical + business + pending contributions add up to the headline change."""
    import re as _re
    eng = _engine()
    a = eng.ask("Why did success rate drop last week?", "head_of_payments", "rules")
    split = next(f["text"] for f in a.facts if f["text"].startswith("By outcome type"))
    parts = [float(x) for x in _re.findall(r"([+-]\d+\.\d+) pts", split)]
    head = float(_re.search(r"\(([+-]\d+\.\d+) pts\)", a.headline).group(1))
    assert len(parts) == 3 and abs(sum(parts) - head) <= 0.02   # 2-dp rounding only


def test_auth_config_problem_is_reported_not_hidden(tmp_path):
    from analyst.auth import config_problem
    assert config_problem() is None                                    # shipped demo file is valid
    assert "No users file" in config_problem(str(tmp_path / "missing.json"))
    empty = tmp_path / "empty.json"
    empty.write_text('{"users": []}')
    assert "no accounts" in config_problem(str(empty))


def test_spelling_is_corrected_and_disclosed():
    from analyst.rule_planner import correct_spelling
    eng = _engine()
    fixed, fixes = correct_spelling("faliure rate by bnak last weak", eng.known_names)
    assert fixed == "failure rate by bank last week" and len(fixes) == 3
    a = eng.ask("Why did sucess rate drop last week?", "head_of_payments", "rules")
    assert a.status == "answered" and "Success rate fell" in a.headline
    assert any("'sucess' as 'success'" in x for x in a.assumptions)
    # a genuinely unknown word is not "corrected" into something else: it still gets a question
    assert correct_spelling("Success rate for premium customers", eng.known_names)[1] == []
    assert eng.ask("Success rate for premium customers last week", "head_of_payments", "rules").status == "clarify"


def test_multi_part_questions_are_split_and_answered_separately():
    from analyst.rule_planner import split_question
    # stays whole: one subject, or 'and' inside a condition or a list
    for q in ("GMV and attempts by bank last week", "Success rate by bank and method last week",
              "Success rate between 9am and 5pm last week", "Why did success rate drop last week?"):
        assert split_question(q) == [q], q
    eng = _engine()
    a, b = eng.ask_parts("Success rate by bank last week and GMV by category in October", "head_of_payments", "rules")
    assert (a.plan["metric"], a.plan["dimensions"]) == ("success_rate", ["bank"])
    assert (b.plan["metric"], b.plan["dimensions"], b.plan["window"]["start"]) == ("gmv", ["category"], "2026-10-01")
    c, d = eng.ask_parts("Why did sucess rate drop last week and what about failure rate?", "head_of_payments", "rules")
    assert c.plan["intent"] == "diagnose_change" and d.plan["metric"] == "failure_rate"
    assert d.plan["window"] == c.plan["window"]                       # part 2 builds on part 1
    assert any("'sucess' as 'success'" in x for x in c.assumptions)  # correction still shown


def test_env_file_loads_but_never_overrides(tmp_path, monkeypatch):
    from analyst.config import load_env
    f = tmp_path / ".env"
    f.write_bytes("﻿# comment\nLLM_PROVIDER=gemini\nGEMINI_API_KEY=\"abc\"  # key\nexport LLM_MODEL=x\n\nBAD LINE\n".encode("utf-8"))
    for k in ("LLM_PROVIDER", "GEMINI_API_KEY", "LLM_MODEL"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("LLM_MODEL", "already-set")
    loaded = load_env(str(f))
    assert sorted(loaded) == ["GEMINI_API_KEY", "LLM_PROVIDER"]
    assert os.environ["GEMINI_API_KEY"] == "abc" and os.environ["LLM_MODEL"] == "already-set"


def test_llm_sql_inclusive_end_date_is_fixed_and_disclosed():
    from analyst.llm import LLMResult
    from analyst.plan import Plan
    eng = _engine()
    eng.llm.provider = "gemini"
    drops_sunday = Plan(intent="free_sql", sql=(
        "SELECT SUM(CASE WHEN status = 'SUCCESS' THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS success_rate_weekends "
        "FROM transactions WHERE created_at >= '2026-10-19' AND created_at <= '2026-10-25' "
        "AND (EXTRACT(DOW FROM created_at) = 0 OR EXTRACT(DOW FROM created_at) = 6)"))
    eng.llm.plan = lambda q, **kw: LLMResult(plan=drops_sunday)
    a = eng.ask("What was the success rate on weekends last week?", "head_of_payments", "llm")
    truth = eng.ex.raw("""SELECT AVG((status='SUCCESS')::INT) AS r FROM raw.transactions_clean
        WHERE created_at >= '2026-10-24' AND created_at < '2026-10-26'""").r[0]
    assert a.status == "answered" and f"{truth:.2%}" in a.headline          # 84.97%, not Saturday-only
    assert any("last day is included" in x for x in a.assumptions)


def test_spelling_leaves_real_words_alone():
    from analyst.rule_planner import correct_spelling
    eng = _engine()
    assert correct_spelling("how did sucess if defined here?", eng.known_names)[1] == [("sucess", "success")]
    assert correct_spelling("most failurs last week", eng.known_names)[1] == [("failurs", "failures")]


def test_every_chart_kind_renders():
    from analyst import charts
    eng = _engine()
    kinds = set()
    for q in ("Why did success rate drop last week?", "Success rate by bank last week", "Show daily success rate trend",
              "Do settlements agree with transactions on success rate last week?",
              "Success rate by bank and method last week"):
        for spec in eng.ask(q, "head_of_payments", "rules").charts:
            charts.render(spec, [])
            kinds.add(spec["kind"])
    assert kinds == {"trend", "contrib", "bar", "sources"}


def test_role_capabilities_are_enforced_by_the_engine():
    from analyst.rbac import ROLES
    eng = _engine()
    # SQL console: Head of Payments and Risk Ops yes, KAM no (refused by the engine, not only hidden)
    assert eng.run_sql("SELECT COUNT(*) AS n FROM transactions", "head_of_payments").status == "answered"
    assert eng.run_sql("SELECT COUNT(*) AS n FROM transactions", "risk_ops").status == "answered"
    kam = eng.run_sql("SELECT COUNT(*) AS n FROM transactions", "kam_portfolio_a")
    assert kam.status == "denied" and kam.denial_rule == "sql_console"
    # executed SQL (physical tables + scope rewrite): only Risk Ops receives it
    q = "Success rate by bank last week"
    for role, visible in (("risk_ops", True), ("head_of_payments", False), ("kam_portfolio_a", False)):
        a = eng.ask(q, role, "rules")
        assert a.status == "answered" and all(bool(r.executable_sql) == visible for r in a.queries), role
    # the table shown to users matches the enforced rules
    caps = {r: {name: ok for name, ok, _ in ROLES[r].capabilities()} for r in ROLES}
    assert caps["kam_portfolio_a"] == {"Ask questions": True, "Raw ingested data": False, "Customer PII": False,
                                       "SQL console": False, "Executed SQL": False}
    assert caps["risk_ops"]["Customer PII"] and caps["risk_ops"]["Executed SQL"]
    assert caps["head_of_payments"]["SQL console"] and not caps["head_of_payments"]["Customer PII"]


def test_audit_log_keeps_executed_sql_even_when_hidden_from_the_role(tmp_path):
    from analyst.engine import AnalystEngine
    db = os.path.join(os.path.dirname(__file__), "..", "data", "payments.duckdb")
    eng = AnalystEngine(db, audit_path=str(tmp_path / "audit.jsonl"))
    eng.ask("Success rate by merchant last week", "kam_portfolio_a", "rules", user="kam.a")
    rows = eng.ex.read_audit(user="kam.a")
    assert rows and all("merchant_id IN" in r["executable_sql"] for r in rows if r["status"] == "ok")


def test_suggestions_only_offer_what_the_role_can_run():
    from analyst.rbac import ROLES
    from ui.pages import suggestions_for
    eng = _engine()
    for role_key, role in ROLES.items():
        for q in suggestions_for(role):
            a = eng.ask(q, role_key, "rules")
            assert a.status != "denied", (role_key, q, a.message)


def test_sweep_regressions_dates_metrics_names():
    """Questions that used to get a confident answer to a different question."""
    eng = _engine()
    ask = lambda q: eng.ask(q, "head_of_payments", "rules")  # noqa: E731
    for q, start, end in (("success rate on 14 Oct", "2026-10-14", "2026-10-14"),
                          ("success rate on 2026-10-14", "2026-10-14", "2026-10-14"),
                          ("success rate from 1 oct to 10 oct", "2026-10-01", "2026-10-10"),
                          ("success rate 1-10 october", "2026-10-01", "2026-10-10"),
                          ("success rate for the first week of october", "2026-10-01", "2026-10-07"),
                          ("success rate today", "2026-10-25", "2026-10-25")):
        p = ask(q).plan
        assert (p["window"]["start"], p["window"]["end"]) == (start, end), q
    a = ask("how many transactions failed yesterday")
    assert a.plan["metric"] == "failures" and "841" in a.headline
    assert ask("success rate by bank tier").plan["dimensions"] == ["bank_tier"]
    assert ask("success rate by week").plan["dimensions"] == ["week"]
    assert ask("success rate trend by method").charts[0]["kind"] == "trend_multi"
    assert ask("top failure reasons last week").plan["metric"] == "failures"
    assert ask("Success rate for delta coop bank last week").plan["filters"] == {"bank": ["Delta Co-op Bank"]}
    assert ask("Why has the success rate gone down recently?").plan["intent"] == "diagnose_change"
    assert ask("success rate last month").plan["baseline"] is None      # no 1-day 'previous month'
    assert "lowest UPI" not in ask("success rate for upi").headline      # one segment is not ranked
    assert "GMV (INR)" in ask("daily gmv trend").headline
