"""Renders one engine Answer. Presentation only: everything shown comes from the Answer object."""
from __future__ import annotations

import re

import pandas as pd
import streamlit as st

from analyst import charts

from . import components as c

_STATUS = {  # how non-answers are presented
    "clarify": ("info", "Needs clarification"),
    "denied": ("bad", "Access denied"),
    "blocked": ("bad", "Blocked by the SQL safety guard"),
    "error": ("warn", "Could not complete this question"),
}


def _planner_label(a) -> str:
    m = re.match(r"LLM \((\w+):([^)]+)\)", a.planner_used or "")
    if m:
        return f"LLM · {m.group(2)}" + (" · custom SQL" if "free SQL" in a.planner_used else "")
    return "Rules" + (" (LLM unavailable)" if any(s.status == "fallback" for s in a.steps) else "")


def _table(df: pd.DataFrame, key: str) -> None:
    """Dataframe with sensible number formats: rates as %, money and counts with separators."""
    cfg = {}
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        name = str(col).lower()
        s = df[col].dropna()
        if re.search(r"rate|share|ratio", name) and len(s) and s.between(-1, 1).all():
            cfg[col] = st.column_config.NumberColumn(format="percent")
        elif "gmv" in name or "amount" in name or "ticket" in name:
            cfg[col] = st.column_config.NumberColumn(format="₹%,.0f")
        elif pd.api.types.is_integer_dtype(df[col]) or name in ("attempts", "denom", "numer"):
            cfg[col] = st.column_config.NumberColumn(format="%,d")
    st.dataframe(df, width="stretch", hide_index=True, column_config=cfg, key=key)


def render(a, dq_days: list, key: str = "a") -> None:
    spelling = next((x for x in a.steps if x.name == "spelling"), None)
    if spelling:
        st.caption(f":material/spellcheck: {spelling.detail} Answered: *{a.question}*")

    if a.status != "answered":
        kind, title = _STATUS.get(a.status, ("warn", a.status.title()))
        text = a.message + (f"  (rule: {a.denial_rule})" if a.denial_rule else "")
        c.callout(kind, text, title)
        if a.status == "clarify" and a.clarification and a.clarification.get("options"):
            opts = a.clarification["options"]
            cols = st.columns(min(len(opts), 4))
            for i, opt in enumerate(opts):
                if cols[i % len(cols)].button(opt, key=f"{key}_opt_{i}", width="stretch"):
                    standalone = a.clarification.get("standalone") or a.message.startswith("I couldn't")
                    base = "" if standalone else a.question
                    st.session_state.pending_q = (base + " " + opt).strip() if base else opt
                    st.rerun()
        if a.caveats:
            for cv in a.caveats:
                c.callout("warn", cv)
        with st.expander("Trace", icon=":material/account_tree:"):
            st.dataframe(pd.DataFrame([vars(s) for s in a.steps]), width="stretch", hide_index=True)
        return

    c.headline(a.headline, c.confidence_pill(a.confidence.get("level", "")))
    cost = sum(x.cost_usd for x in a.llm_calls)
    if not a.llm_calls:
        cost_txt = "none"
    elif all(x.cached for x in a.llm_calls):
        cost_txt = "cached plan"
    else:
        cost_txt = f"${cost:.4f}" if cost >= 0.0001 else "<$0.0001"
    c.statline([("Planner", _planner_label(a)), ("Queries", str(len(a.queries))),
                ("Time", f"{a.total_ms:,.0f} ms"), ("Model cost", cost_txt)])

    t_sum, t_chart = st.tabs([":material/article: Summary", ":material/monitoring: Charts"])

    with t_sum:
        c.section("Findings", len(a.facts))
        st.caption("Computed from the data. The reference after each finding (Q1, Q2, ...) is the query it "
                   "came from: open Evidence below to see it.")
        c.item_list([f["text"] for f in a.facts], "facts", [f["evidence"] for f in a.facts])
        if a.hypotheses:
            c.section("Likely causes", len(a.hypotheses))
            st.caption("Consistent with the findings, but not proven by this data.")
            c.item_list(a.hypotheses, "hypos")
        if a.recommendations:
            c.section("Recommended actions", len(a.recommendations))
            c.item_list(a.recommendations, "recs")
        if a.caveats:
            c.section("Caveats", len(a.caveats))
            for cv in a.caveats:
                c.callout("warn", cv)
        c.section("Method")
        cit = a.citation or {}
        c.kv([("Dataset", cit.get("datasets", "")), ("Metric", cit.get("metric", "")),
              ("Period", cit.get("window", "")), ("Compared with", cit.get("baseline", "")),
              ("Filters", "none" if cit.get("filters") in (None, "none", {}) else str(cit.get("filters"))),
              ("Planner", a.planner_used)])
        if a.assumptions:
            st.markdown("**Assumptions**\n" + "\n".join(f"- {x}" for x in a.assumptions))
        if a.confidence.get("reasons"):
            st.markdown("**Why this confidence**\n" + "\n".join(f"- {x}" for x in a.confidence["reasons"]))

    with t_chart:
        if not a.charts:
            c.empty_state("No chart for this answer", "Charts appear for trends, breakdowns and diagnoses.")
        for i, spec in enumerate(a.charts):
            st.plotly_chart(charts.render(spec, dq_days), width="stretch", key=f"{key}_chart_{i}",
                            config={"displaylogo": False})
            if spec["kind"] == "contrib":
                st.caption("Mix effect: traffic moved toward or away from the segment. "
                           "Rate effect: the segment's own rate changed.")

    # Evidence: the proof behind the answer, one click away but not in the reader's way
    n_tables = len(a.tables)
    label = (f"Evidence · {len(a.queries)} {'query' if len(a.queries) == 1 else 'queries'} · "
             f"{n_tables} {'table' if n_tables == 1 else 'tables'} · trace")
    with st.expander(label, icon=":material/fact_check:", expanded=False):
        e_sql, e_data, e_trace = st.tabs([":material/code: Queries", ":material/table: Data",
                                          ":material/account_tree: Trace"])
        with e_sql:
            st.caption("Every query passed the SQL safety guard before it ran.")
            for q in a.queries:
                label = (f"{q.query_id} · {q.purpose} · {q.rows:,} rows · {q.ms:,.0f} ms"
                         + (" · cached" if q.cached else ""))
                with st.expander(label):
                    st.code(q.sql, language="sql")
                    st.caption("Guard checks passed: " + " · ".join(q.checks))
                    if q.executable_sql:  # only roles allowed to see internals receive it from the engine
                        with st.popover("Executed SQL (after access-scope rewrite)", icon=":material/shield:"):
                            st.code(q.executable_sql, language="sql")
        with e_data:
            if not a.tables:
                c.empty_state("No tables", "This answer did not produce tabular results.")
            for i, (name, df) in enumerate(a.tables.items()):
                st.markdown(f"**{name}**")
                _table(df, key=f"{key}_table_{i}")
        with e_trace:
            st.markdown("**Workflow steps**")
            st.dataframe(pd.DataFrame([vars(s) for s in a.steps]), width="stretch", hide_index=True)
            if a.llm_calls:
                st.markdown("**Model calls**")
                st.dataframe(pd.DataFrame([vars(x) for x in a.llm_calls]), width="stretch", hide_index=True)
            if a.plan:
                st.markdown("**Analysis plan**")
                st.json(a.plan, expanded=False)
