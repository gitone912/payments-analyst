"""
Streamlit UI for the Payments AI Business Analyst.
Run:  streamlit run app.py
"""
from __future__ import annotations

import os
import sys

import pandas as pd
import streamlit as st

from analyst import charts
from analyst.auth import auth_mode, config_problem, failure_reason, user_from_headers, verify_password
from analyst.engine import AnalystEngine
from analyst.rbac import ROLES

DB = os.path.join(os.path.dirname(__file__), "data", "payments.duckdb")

st.set_page_config(page_title="Payments AI Analyst", page_icon="📊", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1.6rem; max-width: 1250px;}
.fact {border-left: 3px solid #2a78d6; padding: 6px 12px; margin: 6px 0; background: #f6f8fc; border-radius: 4px;}
.hypo {border-left: 3px solid #eb6834; padding: 6px 12px; margin: 6px 0; background: #fdf6f2; border-radius: 4px;}
.rec  {border-left: 3px solid #1baf7a; padding: 6px 12px; margin: 6px 0; background: #f2faf6; border-radius: 4px;}
.ev {font-family: ui-monospace, monospace; font-size: 0.78rem; color: #52514e; background: #ecebe7;
     padding: 1px 6px; border-radius: 8px; margin-left: 6px;}
.headline {font-size: 1.25rem; font-weight: 600; margin: 0.4rem 0 0.8rem 0;}
.small {color: #52514e; font-size: 0.85rem;}
</style>""", unsafe_allow_html=True)


@st.cache_resource
def get_engine() -> AnalystEngine:
    if not os.path.exists(DB):
        from data.generate_data import main as gen
        gen(DB)
    return AnalystEngine(DB)


eng = get_engine()

EXAMPLES = [
    "Why did success rate drop last week?",
    "Why did success rate drop?",
    "Success rate by bank last week",
    "Compare UPI vs Card success rate in October",
    "Show daily success rate trend",
    "Do settlements agree with transactions on success rate last week?",
    "Show customer VPAs with the most failures last week",
    "Success rate for QuickTix last week",
    "Success rate last week raw including duplicates",
]

# ------------------------------------------------------------------ authentication
# The role comes from the authenticated account, never from a control the user can change.
def current_user():
    if auth_mode() == "header":
        return user_from_headers(st.context.headers)
    return st.session_state.get("user")


user = current_user()
if user is None:
    st.title("Payments AI Analyst")
    if auth_mode() == "header":
        st.error("Not authenticated: no trusted identity header from the SSO proxy, or this identity has "
                 "no role assigned. Contact the data platform team.")
        st.stop()
    problem = config_problem()
    if problem:  # a setup error must not look like a wrong password
        st.error(f"**Sign-in is not configured:** {problem}  \nRun `python -m analyst.auth check` for details.")
    with st.form("login"):
        st.subheader("Sign in")
        # autocomplete hints tell the browser this is a sign-in form (not "create a new password")
        username = st.text_input("Username", autocomplete="username")
        password = st.text_input("Password", type="password", autocomplete="current-password")
        if st.form_submit_button("Sign in", type="primary"):
            u = verify_password(username, password)
            if u is None and (not username.strip() or not password):
                # a browser-autofilled value can be visible but never sent to the app
                st.error("Username or password was empty when the form was sent. If your browser filled "
                         "them in, click into each field and type the value, then sign in again.")
                print(f"[auth] sign-in failed: {failure_reason(username, password)}", file=sys.stderr)
            elif u is None:
                st.error("Invalid username or password.")
                print(f"[auth] sign-in failed: {failure_reason(username, password)}", file=sys.stderr)
            else:
                st.session_state.clear()
                st.session_state.user = u
                st.rerun()
    st.caption("Demo accounts are listed in the README (one per role).")
    st.stop()

role_key = user.role

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Session")
    st.markdown(f"**{user.display_name}**  \n`{user.username}` · {ROLES[role_key].label}")
    st.caption(ROLES[role_key].description)
    if auth_mode() != "header" and st.button("Sign out"):
        st.session_state.clear()
        st.rerun()
    planner_mode = st.radio("Planner", ["auto", "rules", "llm"], horizontal=True,
                            help="auto = LLM if configured, deterministic planner as fallback")
    if eng.llm.enabled:
        st.success(f"LLM: {eng.llm.provider} / {eng.llm.model}")
    else:
        st.info("LLM: not configured (set LLM_PROVIDER + key). Using deterministic planner.")
    # per-session demo switch; passed with each request, never set on the shared engine
    simulate_outage = st.toggle("Simulate model-provider outage", value=False)

    st.divider()
    st.subheader("Data quality")
    for f in eng.dq:
        icon = {"pass": "✅", "warn": "⚠️", "fail": "⛔"}[f.status]
        with st.expander(f"{icon} {f.check.replace('_', ' ')}"):
            st.write(f.summary)
            if f.handling:
                st.caption("Handling: " + f.handling)
    st.caption(f"Data: {eng.data_start:%d %b} - {eng.data_end:%d %b %Y}")

    st.divider()
    st.subheader("Try")
    for q in EXAMPLES:
        if st.button(q, width="stretch"):
            st.session_state.pending_q = q

# ------------------------------------------------------------------ main
st.title("Payments AI Analyst")
st.caption("Ask about payment performance. Every number comes from an executed, guarded query - "
           "see the Evidence tab.")

tab_ask, tab_sql, tab_log = st.tabs(["Ask a question", "SQL playground", "Session log"])


def render_answer(a):
    status_box = {"denied": st.error, "blocked": st.error, "error": st.warning, "clarify": st.info}
    if a.status in status_box:
        label = {"denied": "🔒 Access denied", "blocked": "🛡️ Blocked by safety guard",
                 "error": "Error", "clarify": "❓ Need a clarification"}[a.status]
        status_box[a.status](f"**{label}** - {a.message}" + (f"  \n`rule: {a.denial_rule}`" if a.denial_rule else ""))
        if a.status == "clarify" and a.clarification:
            cols = st.columns(len(a.clarification.get("options", [])) or 1)
            for c, opt in zip(cols, a.clarification.get("options", [])):
                if c.button(opt, key=f"opt_{opt}"):
                    base = a.question if not a.message.startswith("I couldn't") else ""
                    st.session_state.pending_q = (base + " " + opt).strip() if base else opt
                    st.rerun()
        if a.status != "answered":
            with st.expander("Trace"):
                for s in a.steps:
                    st.write(f"`{s.name}` **{s.status}** - {s.detail}")
            return

    if a.headline:
        st.markdown(f"<div class='headline'>{a.headline}</div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    lvl = a.confidence.get("level", "-")
    c1.metric("Confidence", lvl.title())
    c2.metric("Queries run", len(a.queries))
    c3.metric("Latency", f"{a.total_ms:,.0f} ms")
    cost = sum(c.cost_usd for c in a.llm_calls)
    c4.metric("Model cost", f"${cost:.4f}" if a.llm_calls else "$0 (no LLM)")

    t1, t2, t3, t4, t5 = st.tabs(["Answer", "Charts", "Evidence (SQL)", "Tables", "Trace"])
    with t1:
        st.markdown("#### Observed facts")
        st.caption("Directly computed from the data. Chips link to the query in the Evidence tab.")
        for f in a.facts:
            chips = "".join(f"<span class='ev'>{e}</span>" for e in f["evidence"])
            st.markdown(f"<div class='fact'>{f['text']}{chips}</div>", unsafe_allow_html=True)
        if a.hypotheses:
            st.markdown("#### Hypotheses")
            st.caption("Plausible explanations consistent with the facts - not proven by this data.")
            for h in a.hypotheses:
                st.markdown(f"<div class='hypo'>{h}</div>", unsafe_allow_html=True)
        if a.recommendations:
            st.markdown("#### Recommended next steps")
            for r in a.recommendations:
                st.markdown(f"<div class='rec'>{r}</div>", unsafe_allow_html=True)
        if a.caveats:
            st.markdown("#### Caveats")
            for c in a.caveats:
                st.warning(c)
        st.markdown("#### Basis of this answer")
        cit = a.citation or {}
        st.markdown(
            f"- **Dataset:** {cit.get('datasets', '-')}\n- **Metric:** {cit.get('metric', '-')}\n"
            f"- **Window:** {cit.get('window', '-')}" + (f"  vs baseline **{cit['baseline']}**" if cit.get('baseline') else "")
            + f"\n- **Filters:** {cit.get('filters', 'none')}\n- **Planner:** {a.planner_used}")
        if a.assumptions:
            st.markdown("**Assumptions made:**\n" + "\n".join(f"- {x}" for x in a.assumptions))
        if a.confidence.get("reasons"):
            st.markdown("**Why this confidence:**\n" + "\n".join(f"- {x}" for x in a.confidence["reasons"]))
    with t2:
        dq_days = sorted({d for f in eng.dq if f.status != "pass" and f.check != "source_disagreement"
                          for d in f.affected_days})
        if not a.charts:
            st.caption("No chart for this answer.")
        for spec in a.charts:
            st.plotly_chart(charts.render(spec, dq_days), width="stretch")
            if spec["kind"] == "contrib":
                st.caption("Blue = share of traffic moved toward/away from the segment. "
                           "Orange = the segment's own success rate changed.")
    with t3:
        for q in a.queries:
            with st.expander(f"{q.query_id} · {q.purpose} · {q.rows} rows · {q.ms:.0f} ms"
                             + (" · cached" if q.cached else "")):
                st.code(q.sql, language="sql")
                st.caption("Guard checks passed: " + " · ".join(q.checks))
                with st.popover("Executed SQL (after scope rewrite)"):
                    st.code(q.executable_sql, language="sql")
    with t4:
        for name, df in a.tables.items():
            st.markdown(f"**{name}**")
            st.dataframe(df, width="stretch", hide_index=True)
    with t5:
        st.markdown("**Workflow steps**")
        st.dataframe(pd.DataFrame([vars(s) for s in a.steps]), width="stretch", hide_index=True)
        if a.llm_calls:
            st.markdown("**Model calls**")
            st.dataframe(pd.DataFrame([vars(c) for c in a.llm_calls]), width="stretch", hide_index=True)
        if a.plan:
            st.markdown("**Analysis plan**")
            st.json(a.plan)


with tab_ask:
    pending = st.session_state.pop("pending_q", None)
    url_q = st.query_params.get("q")
    if url_q and not st.session_state.get("url_q_done"):
        st.session_state.url_q_done = True
        pending = url_q
    if pending:
        st.session_state.q = pending
    q = st.text_input("Question", key="q", placeholder="e.g. Why did success rate drop last week?")
    prev = st.session_state.get("prev")
    if prev and prev["role"] == role_key:
        st.caption(f"Follow-ups such as \"now break that down by bank\" or \"what about UPI?\" build on: "
                   f"*{prev['question']}*")
    clicked = st.button("Analyse", type="primary") or bool(pending)
    if clicked and q.strip():
        with st.spinner("Planning, validating and querying..."):
            context = prev if prev and prev["role"] == role_key else None
            ans = eng.ask(q, role_key, planner_mode, previous=context, user=user.username,
                          simulate_outage=simulate_outage)
            st.session_state.last = ans
            if ans.status == "answered" and ans.plan:
                st.session_state.prev = {"question": q, "plan": ans.plan, "role": role_key}
    if "last" in st.session_state:
        render_answer(st.session_state.last)

with tab_sql:
    st.caption("Hand-written SQL goes through the same guard and role scope as model-written SQL. "
               "Tables: transactions, transactions_raw, merchants, banks, payment_methods, settlements.")
    presets = {
        "Safe aggregate": "SELECT pm.method_name, COUNT(*) AS attempts,\n  AVG(CASE WHEN t.status='SUCCESS' THEN 1.0 ELSE 0 END) AS sr\n"
                          "FROM transactions t JOIN payment_methods pm ON t.method_id = pm.method_id\nGROUP BY 1 ORDER BY 2 DESC",
        "Unsafe: DELETE": "DELETE FROM transactions WHERE status = 'FAILED'",
        "Unsafe: file read": "SELECT * FROM read_csv_auto('/etc/passwd')",
        "Expensive: cartesian": "SELECT COUNT(*) FROM transactions a, transactions b",
        "PII column": "SELECT customer_vpa, COUNT(*) FROM transactions WHERE status='FAILED' GROUP BY 1 ORDER BY 2 DESC",
        "Raw schema bypass": "SELECT COUNT(*) FROM raw.transactions",
    }
    pick = st.selectbox("Preset", list(presets))
    sql = st.text_area("SQL", presets[pick], height=150)
    if st.button("Run through guard"):
        st.session_state.sql_ans = eng.run_sql(sql, role_key, user=user.username)
    if "sql_ans" in st.session_state:
        a = st.session_state.sql_ans
        if a.status != "answered":
            st.error(f"**{a.status.upper()}** - {a.message}" + (f"  \n`rule: {a.denial_rule}`" if a.denial_rule else ""))
        else:
            q0 = a.queries[-1]
            st.success(f"Passed guard · {q0.rows} rows · {q0.ms:.0f} ms")
            st.caption(" · ".join(q0.checks))
            st.dataframe(a.tables["Result"], width="stretch", hide_index=True)

with tab_log:
    st.caption("Your queries from the persistent audit log (who, role, when, outcome). "
               "You only see your own; the full log is for platform admins.")
    rows = eng.ex.read_audit(user=user.username)
    if rows:
        st.dataframe(pd.DataFrame(rows[::-1]).drop(columns=["executable_sql"], errors="ignore"),
                     width="stretch", hide_index=True)
    else:
        st.info("No queries yet.")
