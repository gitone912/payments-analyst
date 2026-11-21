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


SECRET_NAMES: list[str] = []  # names only (never values), shown when the LLM is off on a deployment


def _secrets_to_env() -> None:
    """On Streamlit Community Cloud the LLM key is set in the app's Secrets screen, not a .env file.
    Copy secrets into the environment: top level or inside a [section], any letter case
    ('gemini_api_key' -> GEMINI_API_KEY). Values already set in the environment win."""
    try:
        items = list(st.secrets.items())
    except Exception:  # noqa: BLE001  no secrets file: local run, .env is used instead
        return
    # names this app itself copied from secrets earlier: those follow later edits of the Secrets screen;
    # anything set some other way (terminal, platform) is left untouched
    ours = set(filter(None, os.environ.get("_PA_FROM_SECRETS", "").split(",")))
    for key, value in items:
        pairs = value.items() if hasattr(value, "items") else [(key, value)]
        for k, v in pairs:
            if isinstance(v, (str, int, float, bool)):
                SECRET_NAMES.append(str(k))
                name = str(k).upper()
                if name in ours or name not in os.environ:
                    os.environ[name] = str(v).strip()
                    ours.add(name)
    os.environ["_PA_FROM_SECRETS"] = ",".join(sorted(ours))


_secrets_to_env()
load_env()  # .env in the project folder (LLM key etc.); must run before the engine is built

from analyst.auth import auth_mode, user_from_headers  # noqa: E402
from analyst.engine import AnalystEngine  # noqa: E402
from analyst.rbac import ROLES  # noqa: E402
from ui import pages, theme  # noqa: E402

DB = os.path.join(os.path.dirname(__file__), "data", "payments.duckdb")

st.set_page_config(page_title="Payments Analyst", page_icon="assets/logo_icon.svg", layout="wide",
                   initial_sidebar_state="expanded")
theme.apply()


LLM_SETTINGS = ("LLM_PROVIDER", "GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL",
                "LLM_MODEL", "LLM_FALLBACK_MODELS")


def _llm_fingerprint() -> str:
    """One-way hash of the LLM settings. The engine is cached per fingerprint, so if the settings
    change while the server runs (e.g. Streamlit Cloud secrets arriving after start-up), a new engine
    with the new settings is built instead of the old one being reused."""
    import hashlib
    return hashlib.sha256("|".join(os.getenv(k, "") for k in LLM_SETTINGS).encode()).hexdigest()[:16]


@st.cache_resource(max_entries=2)
def get_engine(llm_fingerprint: str) -> AnalystEngine:
    if not os.path.exists(DB):
        from data.generate_data import main as gen
        gen(DB)
    return AnalystEngine(DB)


eng = get_engine(_llm_fingerprint())


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
role = ROLES[user.role]
nav = st.navigation([
    st.Page(lambda: pages.ask_page(ctx), title="Ask", icon=":material/chat:", url_path="ask", default=True),
    # the SQL console is only listed for roles allowed to run hand-written SQL; for others the address
    # still resolves (hidden), showing "not available for your role" rather than "page not found".
    # The engine refuses their SQL either way.
    st.Page(lambda: pages.sql_page(ctx), title="SQL console", icon=":material/terminal:", url_path="sql",
            visibility="visible" if role.can_run_sql else "hidden"),
    st.Page(lambda: pages.audit_page(ctx), title="Audit log", icon=":material/history:", url_path="audit"),
], position="sidebar")
nav.run()
