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
