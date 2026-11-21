"""
Design system: tokens and the one stylesheet for the app's own elements.

Colours, fonts, radius and the light/dark palettes live in .streamlit/config.toml (native theming).
This file only styles elements the app renders itself (classes prefixed `pa-`) and keyed containers
(`st-key-*`, a documented Streamlit hook). It never targets Streamlit's internal DOM, so it survives
Streamlit upgrades. Colours here are translucent or `currentColor`, so they read in both themes.
"""
from __future__ import annotations

import streamlit as st

ACCENT = "#4F46E5"
STATUS = {  # semantic colours: used only where they mean something
    "ok": "#10B981", "warn": "#F59E0B", "bad": "#EF4444", "info": "#6366F1", "muted": "#94A3B8",
}
CHART = {"primary": "#4F46E5", "secondary": "#F97316", "tertiary": "#10B981", "neutral": "#94A3B8",
         "critical": "#EF4444", "grid": "rgba(148,163,184,0.22)", "axis": "#94A3B8"}

_CSS = """
<style>
/* ---------- layout ---------- */
.block-container { max-width: 1180px; }

/* ---------- typography helpers ---------- */
.pa-eyebrow { font-size: 0.72rem; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase;
              opacity: 0.55; margin: 0 0 0.35rem 0; }
.pa-muted { opacity: 0.62; }
.pa-num { font-variant-numeric: tabular-nums; }

/* ---------- page header ---------- */
.pa-header { display: flex; align-items: flex-end; justify-content: space-between; gap: 1rem;
             padding-bottom: 0.9rem; margin-bottom: 1.1rem; border-bottom: 1px solid rgba(148,163,184,0.22); }
.pa-header h1 { font-size: 1.375rem; font-weight: 650; margin: 0; padding: 0; line-height: 1.3; }
.pa-header p { margin: 0.2rem 0 0 0; font-size: 0.86rem; opacity: 0.62; }
.pa-header .pa-meta { font-size: 0.78rem; opacity: 0.62; text-align: right; white-space: nowrap; }

/* ---------- brand (sidebar) ---------- */
.pa-brand { display: flex; align-items: center; gap: 0.55rem; margin: 0.15rem 0 0.2rem 0; }
.pa-brand-mark { width: 22px; height: 22px; border-radius: 6px; background: #4F46E5; display: flex;
                 align-items: center; justify-content: center; color: #fff; font-weight: 700; font-size: 0.75rem; }
.pa-brand-name { font-weight: 650; font-size: 0.95rem; letter-spacing: -0.01em; }

/* ---------- user block (sidebar) ---------- */
.pa-user { display: flex; align-items: center; gap: 0.6rem; }
.pa-avatar { width: 30px; height: 30px; border-radius: 50%; background: rgba(99,102,241,0.16); color: #6366F1;
             display: flex; align-items: center; justify-content: center; font-weight: 650; font-size: 0.8rem; flex: none; }
.pa-user-name { font-weight: 600; font-size: 0.86rem; line-height: 1.2; }
.pa-user-role { font-size: 0.76rem; opacity: 0.6; line-height: 1.25; }

/* ---------- status dot list (data health) ---------- */
.pa-dot { display: inline-block; width: 7px; height: 7px; border-radius: 50%; margin-right: 0.5rem;
          vertical-align: middle; }
.pa-health { display: flex; justify-content: space-between; align-items: center; font-size: 0.82rem;
             padding: 0.28rem 0; }
.pa-health span:last-child { opacity: 0.55; font-size: 0.75rem; }

.pa-access { font-size: 0.72rem; font-weight: 600; }
.pa-access.yes { color: #10B981; }
.pa-access.no { color: #94A3B8; }

/* ---------- answer: headline + status line ---------- */
.pa-headline { font-size: 1.12rem; font-weight: 600; line-height: 1.45; margin: 0.25rem 0 0.6rem 0;
               letter-spacing: -0.005em; }
.pa-statline { display: flex; flex-wrap: wrap; gap: 0.35rem 1.4rem; font-size: 0.8rem; margin-bottom: 1.1rem; }
.pa-statline div { display: flex; gap: 0.35rem; align-items: baseline; }
.pa-statline .k { opacity: 0.55; }
.pa-statline .v { font-weight: 600; font-variant-numeric: tabular-nums; }
.pa-pill { display: inline-flex; align-items: center; gap: 0.3rem; font-size: 0.74rem; font-weight: 600;
           padding: 0.12rem 0.5rem; border-radius: 999px; border: 1px solid currentColor; line-height: 1.4; }

/* ---------- findings list ---------- */
.pa-list { margin: 0 0 1.1rem 0; padding: 0; list-style: none; }
.pa-list li { position: relative; padding: 0.55rem 0 0.55rem 0.95rem; line-height: 1.55; font-size: 0.9rem;
              border-bottom: 1px solid rgba(148,163,184,0.16); }
.pa-list li:last-child { border-bottom: none; }
.pa-list li::before { content: ""; position: absolute; left: 0; top: 0.95rem; width: 5px; height: 5px;
                      border-radius: 50%; background: var(--pa-marker, #94A3B8); }
.pa-list.facts { --pa-marker: #4F46E5; }
.pa-list.hypos { --pa-marker: #F97316; }
.pa-list.recs  { --pa-marker: #10B981; }
.pa-ref { font-family: "JetBrains Mono", ui-monospace, monospace; font-size: 0.7rem; padding: 0.05rem 0.38rem;
          margin-left: 0.4rem; border-radius: 4px; background: rgba(148,163,184,0.18); opacity: 0.85;
          white-space: nowrap; }

/* ---------- callouts (caveats, clarifications, denials, errors) ---------- */
.pa-callout { border: 1px solid rgba(148,163,184,0.28); border-left-width: 3px; border-radius: 0.5rem;
              padding: 0.7rem 0.9rem; margin: 0 0 0.6rem 0; font-size: 0.88rem; line-height: 1.5; }
.pa-callout .t { font-weight: 600; display: block; margin-bottom: 0.15rem; }
.pa-callout.info { border-left-color: #6366F1; background: rgba(99,102,241,0.06); }
.pa-callout.warn { border-left-color: #F59E0B; background: rgba(245,158,11,0.07); }
.pa-callout.bad  { border-left-color: #EF4444; background: rgba(239,68,68,0.07); }
.pa-callout.ok   { border-left-color: #10B981; background: rgba(16,185,129,0.07); }
.pa-callout code { font-size: 0.75rem; }

/* ---------- key/value (method section) ---------- */
.pa-kv { display: grid; grid-template-columns: 9rem 1fr; gap: 0.3rem 1rem; font-size: 0.85rem; margin: 0.2rem 0 0.8rem 0; }
.pa-kv .k { opacity: 0.55; }

/* ---------- empty state ---------- */
.pa-empty { padding: 1.6rem 0 0.6rem 0; }
.pa-empty h3 { font-size: 1rem; font-weight: 600; margin: 0 0 0.25rem 0; }
.pa-empty p { font-size: 0.86rem; opacity: 0.62; margin: 0; }

/* ---------- section titles ---------- */
.pa-section { font-size: 0.8rem; font-weight: 600; letter-spacing: 0.02em; margin: 1.1rem 0 0.2rem 0;
              display: flex; align-items: center; gap: 0.5rem; }
.pa-section .n { opacity: 0.5; font-weight: 500; }

/* ---------- login ---------- */
.st-key-login_card { max-width: 400px; margin: 7vh auto 0 auto; }
.pa-login-title { font-size: 1.15rem; font-weight: 650; margin: 0.8rem 0 0.15rem 0; }
.pa-login-sub { font-size: 0.85rem; opacity: 0.62; margin: 0 0 1rem 0; }

/* ---------- responsive ---------- */
@media (max-width: 760px) {
  .pa-header { flex-direction: column; align-items: flex-start; }
  .pa-header .pa-meta { text-align: left; white-space: normal; }
  .pa-kv { grid-template-columns: 1fr; }
  .pa-kv .k { margin-top: 0.4rem; }
}
</style>
"""


def apply() -> None:
    """Inject the stylesheet once per run."""
    st.html(_CSS)
