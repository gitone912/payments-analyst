"""The app's screens. Each page receives a Context with the engine and the signed-in user; all
analysis is delegated to the engine exactly as before."""
from __future__ import annotations

import sys
from dataclasses import dataclass

import pandas as pd
import streamlit as st

from analyst.auth import auth_mode, config_problem, failure_reason, verify_password
from analyst.rbac import ROLES

from . import answer
from . import components as c

def suggestions_for(role) -> list[str]:
    """Only suggest questions this role is allowed to get answered."""
    out = list(SUGGESTIONS)
    if role.merchant_scope_sql is not None:  # portfolio-scoped: use a merchant they own
        out = [q.replace("QuickTix", "DailyBasket") for q in out]
    if not role.can_see_pii:
        out = [q for q in out if "VPA" not in q]
    if "transactions_raw" not in role.allowed_tables:
        out = [q for q in out if "raw" not in q]
    return out


SUGGESTIONS = [
    "Why did success rate drop last week?",
    "Success rate by bank last week",
    "Compare UPI vs Card success rate in October",
    "Show daily success rate trend",
    "Do settlements agree with transactions on success rate last week?",
    "Technical success rate by bank last week",
    "Show customer VPAs with the most failures last week",
    "Success rate for QuickTix last week",
    "Why did success rate drop?",
    "Success rate last week raw including duplicates",
]

SQL_PRESETS = {
    "Safe aggregate": "SELECT pm.method_name, COUNT(*) AS attempts,\n  AVG(CASE WHEN t.status='SUCCESS' THEN 1.0 ELSE 0 END) AS success_rate\n"
                      "FROM transactions t JOIN payment_methods pm ON t.method_id = pm.method_id\nGROUP BY 1 ORDER BY 2 DESC",
    "Unsafe: DELETE": "DELETE FROM transactions WHERE status = 'FAILED'",
    "Unsafe: file read": "SELECT * FROM read_csv_auto('/etc/passwd')",
    "Expensive: cartesian join": "SELECT COUNT(*) FROM transactions a, transactions b",
    "PII column": "SELECT customer_vpa, COUNT(*) FROM transactions WHERE status='FAILED' GROUP BY 1 ORDER BY 2 DESC",
    "Raw schema bypass": "SELECT COUNT(*) FROM raw.transactions",
}

DEMO_ACCOUNTS = pd.DataFrame([
    {"Username": "head.payments", "Password": "demo-head", "Role": "Head of Payments", "Can see": "All merchants, no PII"},
    {"Username": "risk.ops", "Password": "demo-risk", "Role": "Risk Ops", "Can see": "All merchants, PII"},
    {"Username": "kam.a", "Password": "demo-kam", "Role": "Key Account Manager", "Can see": "Portfolio A merchants only"},
])


@dataclass
class Context:
    eng: object
    user: object
    planner_mode: str = "auto"
    simulate_outage: bool = False

    @property
    def role_key(self) -> str:
        return self.user.role

    def dq_days(self) -> list:
        return sorted({d for f in self.eng.dq if f.status != "pass" and f.check != "source_disagreement"
                       for d in f.affected_days})


# ------------------------------------------------------------------ sign-in
def login_page() -> None:
    with st.container(key="login_card"):
        st.logo("assets/logo.svg", icon_image="assets/logo_icon.svg")
        st.html('<div class="pa-brand-mark" style="width:32px;height:32px;border-radius:8px;font-size:0.95rem">P</div>'
                '<div class="pa-login-title">Sign in to Payments Analyst</div>'
                '<div class="pa-login-sub">Ask questions about payment performance and get answers backed '
                'by the queries that produced them.</div>')
        if auth_mode() == "header":
            c.callout("bad", "No trusted identity header from the SSO proxy, or this identity has no role "
                             "assigned. Contact the data platform team.", "Not authenticated")
            return
        problem = config_problem()
        if problem:  # a setup error must not look like a wrong password
            c.callout("bad", f"{problem} Run `python -m analyst.auth check` for details.", "Sign-in is not configured")
        with st.form("login", border=True):
            # autocomplete hints tell the browser this is a sign-in form (not "create a new password")
            username = st.text_input("Username", autocomplete="username")
            password = st.text_input("Password", type="password", autocomplete="current-password")
            if st.form_submit_button("Sign in", type="primary", width="stretch"):
                u = verify_password(username, password)
                if u is None and (not username.strip() or not password):
                    # a browser-autofilled value can be visible but never sent to the app
                    c.callout("bad", "Username or password was empty when the form was sent. If your browser "
                                     "filled them in, click into each field and type the value.")
                    print(f"[auth] sign-in failed: {failure_reason(username, password)}", file=sys.stderr)
                elif u is None:
                    c.callout("bad", "Invalid username or password.")
                    print(f"[auth] sign-in failed: {failure_reason(username, password)}", file=sys.stderr)
                else:
                    st.session_state.clear()
                    st.session_state.user = u
                    st.rerun()
        with st.expander("Demo accounts"):
            st.dataframe(DEMO_ACCOUNTS, hide_index=True, width="stretch")


# ------------------------------------------------------------------ sidebar
def sidebar(ctx: Context) -> Context:
    eng, user = ctx.eng, ctx.user
    with st.sidebar:
        # who you are and what you may do, first: the role decides what every answer can contain
        c.user_block(user.display_name, ROLES[user.role].label, user.username)
        st.caption(ROLES[user.role].description)
        c.eyebrow("Your access")
        c.access_rows(ROLES[user.role].capabilities())
        st.divider()
        c.eyebrow("Assistant")
        mode = st.segmented_control("Planner", ["Auto", "Rules", "LLM"], default="Auto", key="planner_choice",
                                    help="Auto uses the LLM when it is configured and falls back to the "
                                         "deterministic planner. Rules never calls a model.")
        ctx.planner_mode = (mode or "Auto").lower()
        if eng.llm.enabled:
            st.caption(f":material/check_circle: LLM on · `{eng.llm.model}`"
                       + (f" · backup `{eng.llm.models[1]}`" if len(eng.llm.models) > 1 else ""))
        else:
            st.caption(":material/do_not_disturb_on: LLM off: copy `.env.example` to `.env`, add your key "
                       "and restart. The deterministic planner answers meanwhile.")
        # per-session demo switch; passed with each request, never set on the shared engine
        ctx.simulate_outage = st.toggle("Simulate LLM outage", value=False, key="simulate_outage",
                                        help="Demonstrates the fallback to the deterministic planner.")

        st.divider()
        c.eyebrow("Data health")
        notes = {"pass": "ok", "warn": "warning", "fail": "issue"}
        st.html("".join(c.health_row(f.check.replace("_", " ").capitalize(), f.status, notes[f.status])
                        for f in eng.dq))
        with st.popover("Details", icon=":material/info:", width="stretch"):
            for f in eng.dq:
                st.markdown(f"**{f.check.replace('_', ' ').capitalize()}**  \n{f.summary}")
                if f.handling:
                    st.caption("Handling: " + f.handling)
        st.caption(f"Data {eng.data_start:%d %b} – {eng.data_end:%d %b %Y}")

        st.divider()
        if auth_mode() != "header" and st.button("Sign out", icon=":material/logout:", type="tertiary"):
            st.session_state.clear()
            st.rerun()
    return ctx


# ------------------------------------------------------------------ ask
def ask_page(ctx: Context) -> None:
    eng = ctx.eng
    c.page_header("Ask a question",
                  "Plain-English questions about payment performance. Every number links to the query behind it.",
                  f"{ROLES[ctx.role_key].label} · {eng.data_start:%d %b} – {eng.data_end:%d %b %Y}")

    pending = st.session_state.pop("pending_q", None)
    url_q = st.query_params.get("q")
    if url_q and not st.session_state.get("url_q_done"):
        st.session_state.url_q_done = True
        pending = url_q
    if pending:
        st.session_state.q = pending

    with st.container(border=True, key="composer"):
        q = st.text_input("Question", key="q", placeholder="e.g. Why did success rate drop last week?",
                          label_visibility="collapsed")
        prev = st.session_state.get("prev")

        def new_conversation():
            # runs before the next rerun, so the question box can be cleared safely
            for k in ("prev", "last"):
                st.session_state.pop(k, None)
            st.session_state.q = ""

        b1, b2, note = st.columns([1.1, 1.9, 6], vertical_alignment="center")
        clicked = b1.button("Analyse", type="primary", icon=":material/arrow_forward:", width="stretch") or bool(pending)
        b2.button("New conversation", icon=":material/add_comment:", on_click=new_conversation, width="stretch",
                  disabled="prev" not in st.session_state and "last" not in st.session_state,
                  help="Forget the previous question and answer")
        if prev and prev["role"] == ctx.role_key:
            note.caption(f"Context: *{prev['question']}*. Follow-ups that refer back (\"break that down by "
                         f"bank\", \"what about UPI?\") use it; other questions start fresh.")

    if clicked and q.strip():
        with st.spinner("Planning, validating and querying…"):
            context = prev if prev and prev["role"] == ctx.role_key else None
            answers = eng.ask_parts(q, ctx.role_key, ctx.planner_mode, previous=context,
                                    user=ctx.user.username, simulate_outage=ctx.simulate_outage)
            st.session_state.last = answers
            done = [x for x in answers if x.status == "answered" and x.plan]
            if done:  # follow-ups build on the last part that was answered
                st.session_state.prev = {"question": done[-1].question, "plan": done[-1].plan, "role": ctx.role_key}
        # the context line and buttons are drawn above: redraw once so they reflect the new answer
        st.rerun()

    if "last" not in st.session_state:
        c.empty_state("Start with a question",
                      "Ask about success rate, failures, GMV or volume by bank, method, merchant or period. "
                      "Or pick one of these:")
        def run_suggestion():
            # a callback runs before the page is redrawn, so the pill can be reset here; resetting it
            # in the page body (after the widget exists) raises StreamlitWidgetAlreadyInstantiatedError
            st.session_state.pending_q = st.session_state.suggestion
            st.session_state.suggestion = None

        st.pills("Suggestions", suggestions_for(ROLES[ctx.role_key]), key="suggestion",
                 label_visibility="collapsed", on_change=run_suggestion)
        return

    answers = st.session_state.last if isinstance(st.session_state.last, list) else [st.session_state.last]
    dq_days = ctx.dq_days()
    if len(answers) == 1:
        answer.render(answers[0], dq_days)
        return
    c.callout("info", f"Your question has {len(answers)} parts, so each is answered separately. "
                      "Later parts build on earlier ones.")
    tabs = st.tabs([f"Part {i}: {x.question[:45]}{'…' if len(x.question) > 45 else ''}"
                    for i, x in enumerate(answers, 1)])
    for i, (tab, x) in enumerate(zip(tabs, answers), 1):
        with tab:
            answer.render(x, dq_days, key=f"p{i}")


# ------------------------------------------------------------------ SQL console
def sql_page(ctx: Context) -> None:
    role = ROLES[ctx.role_key]
    c.page_header("SQL console",
                  "Hand-written, read-only SQL. It goes through the same safety guard and access scope as "
                  "model-written SQL.",
                  "Tables: " + ", ".join(sorted(role.allowed_tables)))
    if not role.can_run_sql:  # not in this role's navigation; also refused by the engine
        c.callout("bad", f"{role.label} cannot run hand-written SQL. Ask a question instead: answers are "
                         "scoped to your merchants automatically.", "Not available for your role")
        return
    st.caption(f"Runs as **{role.label}**: "
               + ("all merchants" if role.merchant_scope_sql is None else "your merchants only")
               + ("; customer PII allowed" if role.can_see_pii else "; customer PII columns refused") + ".")
    pick = st.selectbox("Example", list(SQL_PRESETS), key="sql_preset",
                        help="Try the unsafe examples to see the guard block them.")
    sql = st.text_area("SQL", SQL_PRESETS[pick], height=170, key=f"sql_text_{pick}")
    if st.button("Run through guard", type="primary", icon=":material/play_arrow:"):
        st.session_state.sql_ans = ctx.eng.run_sql(sql, ctx.role_key, user=ctx.user.username)
    a = st.session_state.get("sql_ans")
    if a is None:
        return
    if a.status != "answered":
        title = {"blocked": "Blocked by the SQL safety guard", "denied": "Access denied"}.get(a.status, "Query failed")
        c.callout("bad" if a.status in ("blocked", "denied") else "warn",
                  a.message + (f"  (rule: {a.denial_rule})" if a.denial_rule else ""), title)
        return
    q0 = a.queries[-1]
    c.callout("ok", " · ".join(q0.checks), f"Passed the guard · {q0.rows:,} rows · {q0.ms:,.0f} ms")
    answer._table(a.tables["Result"], key="sql_result")


# ------------------------------------------------------------------ audit log
def audit_page(ctx: Context) -> None:
    c.page_header("Audit log", "Your queries: who ran what, as which role, when, and the outcome.",
                  "Other users' entries are visible to platform admins only")
    rows = ctx.eng.ex.read_audit(user=ctx.user.username)
    if not rows:
        c.empty_state("No queries yet", "Queries you run from Ask or the SQL console appear here.")
        return
    df = pd.DataFrame(rows[::-1]).drop(columns=["executable_sql"], errors="ignore")
    order = [x for x in ["at", "query_id", "status", "purpose", "role", "rows", "ms", "cached", "error", "checks"]
             if x in df.columns]
    df = df[order + [x for x in df.columns if x not in order]]
    st.dataframe(df, width="stretch", hide_index=True, column_config={
        "at": st.column_config.TextColumn("Time (UTC)"), "query_id": "Query", "status": "Status",
        "purpose": st.column_config.TextColumn("Purpose", width="large"), "role": "Role",
        "rows": st.column_config.NumberColumn("Rows", format="%,d"),
        "ms": st.column_config.NumberColumn("ms", format="%,.0f"), "cached": "Cached",
    })
