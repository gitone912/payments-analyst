"""
Payments Analyst: Streamlit UI.
Run:  python run_app.py   (or: python -m streamlit run app.py)

Layout and styling live in ui/ (pages, answer view, components, theme) and .streamlit/config.toml.
This file only wires them together: settings, engine, sign-in, navigation.
"""
from __future__ import annotations

import os

import streamlit as st

from analyst.config import load_env

load_env()  # .env in the project folder (LLM key etc.); must run before the engine is built

from analyst.auth import auth_mode, user_from_headers  # noqa: E402
from analyst.engine import AnalystEngine  # noqa: E402
from ui import pages, theme  # noqa: E402

DB = os.path.join(os.path.dirname(__file__), "data", "payments.duckdb")

st.set_page_config(page_title="Payments Analyst", page_icon="assets/logo_icon.svg", layout="wide",
                   initial_sidebar_state="expanded")
theme.apply()


@st.cache_resource
def get_engine() -> AnalystEngine:
    if not os.path.exists(DB):
        from data.generate_data import main as gen
        gen(DB)
    return AnalystEngine(DB)


eng = get_engine()


def current_user():
    # The role comes from the authenticated account, never from a control the user can change.
    if auth_mode() == "header":
        return user_from_headers(st.context.headers)
    return st.session_state.get("user")


user = current_user()
PAGE_PATHS = ("ask", "sql", "audit")
if user is None:
    # Every app address shows the sign-in page while signed out, so a link like /ask?q=... survives
    # sign-in instead of hitting "page not found", and opens the right page afterwards.
    st.navigation([st.Page(pages.login_page, title="Sign in", default=True)]
                  + [st.Page(pages.login_page, title="Sign in", url_path=path) for path in PAGE_PATHS],
                  position="hidden").run()
    st.stop()

st.logo("assets/logo.svg", icon_image="assets/logo_icon.svg")
ctx = pages.sidebar(pages.Context(eng=eng, user=user))
nav = st.navigation([
    st.Page(lambda: pages.ask_page(ctx), title="Ask", icon=":material/chat:", url_path="ask", default=True),
    st.Page(lambda: pages.sql_page(ctx), title="SQL console", icon=":material/terminal:", url_path="sql"),
    st.Page(lambda: pages.audit_page(ctx), title="Audit log", icon=":material/history:", url_path="audit"),
], position="sidebar")
nav.run()
