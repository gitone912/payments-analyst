"""Small presentational building blocks. Pure rendering: no business logic, no engine calls."""
from __future__ import annotations

import html

import streamlit as st

from .theme import STATUS


def esc(text) -> str:
    """Escape user/engine text before it goes into our own HTML. Facts may contain markdown-style
    `code` spans (applied filters); those are rendered as <code>."""
    out = html.escape(str(text))
    parts = out.split("`")
    return "".join(f"<code>{p}</code>" if i % 2 else p for i, p in enumerate(parts))


def page_header(title: str, subtitle: str = "", meta: str = "") -> None:
    st.html(f"""<div class="pa-header"><div><h1>{esc(title)}</h1>{f"<p>{esc(subtitle)}</p>" if subtitle else ""}</div>
            {f'<div class="pa-meta">{esc(meta)}</div>' if meta else ""}</div>""")


def brand() -> None:
    st.html('<div class="pa-brand"><div class="pa-brand-mark">PA</div>'
            '<div class="pa-brand-name">Payments Analyst</div></div>')


def user_block(display_name: str, role_label: str, username: str) -> None:
    initials = "".join(w[0] for w in display_name.replace("(demo)", "").split()[:2]).upper() or "U"
    st.html(f"""<div class="pa-user"><div class="pa-avatar">{esc(initials)}</div><div>
            <div class="pa-user-name">{esc(display_name)}</div>
            <div class="pa-user-role">{esc(role_label)} · <code>{esc(username)}</code></div></div></div>""")


def eyebrow(text: str) -> None:
    st.html(f'<div class="pa-eyebrow">{esc(text)}</div>')


def section(title: str, count: int | None = None) -> None:
    n = f'<span class="n">{count}</span>' if count is not None else ""
    st.html(f'<div class="pa-section">{esc(title)}{n}</div>')


def callout(kind: str, text: str, title: str = "") -> None:
    """kind: info | warn | bad | ok"""
    t = f'<span class="t">{esc(title)}</span>' if title else ""
    st.html(f'<div class="pa-callout {kind}">{t}{esc(text)}</div>')


def item_list(items: list[str], kind: str, refs: list[list[str]] | None = None) -> None:
    """kind: facts | hypos | recs. refs: evidence query ids per item (shown as small references)."""
    lis = []
    for i, text in enumerate(items):
        r = "".join(f'<span class="pa-ref">{esc(q)}</span>' for q in (refs[i] if refs else []))
        lis.append(f"<li>{esc(text)}{r}</li>")
    st.html(f'<ul class="pa-list {kind}">{"".join(lis)}</ul>')


def statline(pairs: list[tuple[str, str]]) -> None:
    cells = "".join(f'<div><span class="k">{esc(k)}</span><span class="v">{esc(v)}</span></div>' for k, v in pairs)
    st.html(f'<div class="pa-statline">{cells}</div>')


def confidence_pill(level: str) -> str:
    color = {"high": STATUS["ok"], "medium": STATUS["warn"], "low": STATUS["bad"]}.get(level, STATUS["muted"])
    return f'<span class="pa-pill" style="color:{color}">{esc(level.title())} confidence</span>'


def headline(text: str, pill_html: str = "") -> None:
    st.html(f'<div class="pa-headline">{esc(text)}</div>{pill_html}')


def kv(rows: list[tuple[str, str]]) -> None:
    cells = "".join(f'<div class="k">{esc(k)}</div><div>{esc(v)}</div>' for k, v in rows if v)
    st.html(f'<div class="pa-kv">{cells}</div>')


def health_row(label: str, status: str, note: str) -> str:
    color = {"pass": STATUS["ok"], "warn": STATUS["warn"], "fail": STATUS["bad"]}.get(status, STATUS["muted"])
    return (f'<div class="pa-health"><span><span class="pa-dot" style="background:{color}"></span>'
            f'{esc(label)}</span><span>{esc(note)}</span></div>')


def empty_state(title: str, text: str) -> None:
    st.html(f'<div class="pa-empty"><h3>{esc(title)}</h3><p>{esc(text)}</p></div>')


def access_rows(capabilities: list[tuple[str, bool, str]]) -> None:
    """What the signed-in role can and cannot do (mirrors what the engine enforces)."""
    rows = []
    for name, allowed, detail in capabilities:
        mark = ('<span class="pa-access yes">Allowed</span>' if allowed
                else '<span class="pa-access no">Not allowed</span>')
        rows.append(f'<div class="pa-health" title="{esc(detail)}"><span>{esc(name)}</span>{mark}</div>')
    st.html("".join(rows))
