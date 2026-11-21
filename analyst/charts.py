"""Plotly figures for the answer bundle. Colours follow the app's design system (ui/theme.py):
one accent for the primary series, orange only as a contrast series, neutral greys for structure."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

S1, S2, S3 = "#4F46E5", "#F97316", "#10B981"   # accent, contrast, positive (fixed order)
# Neutrals read on light and dark backgrounds; text colour comes from the app theme
INK2, GRID = "#94A3B8", "rgba(148,163,184,0.22)"
CRITICAL = "#EF4444"

LAYOUT = dict(
    height=340, margin=dict(l=8, r=8, t=56, b=8),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, system-ui, sans-serif", size=12),
    title=dict(font=dict(size=14), x=0, xanchor="left", y=0.97),
    hovermode="x unified", hoverlabel=dict(font=dict(family="Inter, system-ui, sans-serif", size=12)),
    legend=dict(orientation="h", y=1.1, x=0, title=None, font=dict(size=11)),
    xaxis=dict(gridcolor=GRID, linecolor=GRID, showgrid=False, ticks="outside", tickcolor=GRID),
    yaxis=dict(gridcolor=GRID, linecolor="rgba(0,0,0,0)", zeroline=False),
)


def _layout(title: str, **overrides) -> dict:
    """Shared layout with this chart's title (left-aligned, small) and any per-chart overrides."""
    return {**LAYOUT, **overrides, "title": {**LAYOUT["title"], "text": title}}


def _pct_axis(fig, metric_is_pct: bool):
    if metric_is_pct:
        fig.update_yaxes(tickformat=".1%")


def trend(spec: dict, dq_days: list | None = None) -> go.Figure:
    df = spec["df"].copy()
    df["day"] = pd.to_datetime(df["day"])
    y = spec["y"]
    is_pct = y in ("success_rate", "success_rate_excl_pending", "failure_rate")
    fig = go.Figure(go.Scatter(x=df.day, y=df[y], mode="lines+markers", name=spec["title"],
                               line=dict(color=S1, width=2), marker=dict(size=5, color=S1,
                               line=dict(width=0)),
                               customdata=df["attempts"],
                               hovertemplate="%{y:.2%}<br>%{customdata:,} attempts<extra></extra>"
                               if is_pct else "%{y:,.0f}<extra></extra>"))
    for name, w in (spec.get("windows") or {}).items():
        fig.add_vrect(x0=pd.Timestamp(w.start) - pd.Timedelta(hours=12),
                      x1=pd.Timestamp(w.end) + pd.Timedelta(hours=12),
                      fillcolor=INK2 if name == "baseline" else S1, opacity=0.07, line_width=0,
                      annotation_text=name, annotation_position="top left",
                      annotation_font_color=INK2)
    if dq_days:
        sel = df[df.day.dt.date.isin(dq_days)]
        if len(sel):
            fig.add_trace(go.Scatter(x=sel.day, y=sel[y], mode="markers", name="Data-quality issue",
                                     marker=dict(size=9, symbol="x", color=CRITICAL),
                                     hovertemplate="data-quality issue on this day<extra></extra>"))
    fig.update_layout(**_layout(spec["title"]))
    fig.update_layout(showlegend=bool(dq_days))
    _pct_axis(fig, is_pct)
    return fig


def contrib(spec: dict, top: int = 10) -> go.Figure:
    df = spec["df"].copy()
    df = df.reindex(df.total_effect.abs().sort_values(ascending=False).index).head(top)
    df = df.iloc[::-1]
    lab = df.segment.astype(str)
    fig = go.Figure()
    fig.add_bar(y=lab, x=df.mix_effect * 100, orientation="h", name="Mix effect (traffic shifted)",
                marker=dict(color=S1, line=dict(width=0), cornerradius=3),
                hovertemplate="%{y}<br>mix %{x:+.2f} pts<extra></extra>")
    fig.add_bar(y=lab, x=df.rate_effect * 100, orientation="h", name="Rate effect (segment got worse/better)",
                marker=dict(color=S2, line=dict(width=0), cornerradius=3),
                hovertemplate="%{y}<br>rate %{x:+.2f} pts<extra></extra>")
    fig.update_layout(**_layout(spec["title"] + " (percentage points)", hovermode="closest",
                                height=90 + 30 * len(df)), barmode="relative")
    fig.update_xaxes(ticksuffix=" pts", zeroline=True, zerolinecolor=INK2)
    return fig


def bar(spec: dict) -> go.Figure:
    df = spec["df"].copy()
    x, y, color = spec["x"], spec["y"], spec.get("color")
    is_pct = y in ("success_rate", "success_rate_excl_pending", "failure_rate")
    df = df.sort_values(y)
    fig = go.Figure()
    if color:
        palette = [S1, S2, S3, "#EAB308"]
        for i, (k, g) in enumerate(df.groupby(color)):
            fig.add_bar(y=g[x].astype(str), x=g[y], orientation="h", name=str(k),
                        marker=dict(color=palette[i % 4], cornerradius=3))
        fig.update_layout(barmode="group")
    else:
        fig.add_bar(y=df[x].astype(str), x=df[y], orientation="h",
                    marker=dict(color=S1, cornerradius=3, line=dict(width=0)),
                    text=[f"{v:.1%}" if is_pct else f"{v:,.0f}" for v in df[y]], textposition="outside",
                    customdata=df["attempts"],
                    hovertemplate="%{y}: %{x:.2%}<br>%{customdata:,} attempts<extra></extra>" if is_pct
                    else "%{y}: %{x:,.0f}<extra></extra>")
    fig.update_layout(**_layout(spec["title"], hovermode="closest", height=90 + 28 * len(df)),
                      showlegend=bool(color))
    if is_pct:
        lo = max(0, df[y].min() - 0.05)
        fig.update_xaxes(tickformat=".0%", range=[lo, min(1, df[y].max() + 0.04)])
    return fig


def sources(spec: dict) -> go.Figure:
    df = spec["df"].copy()
    df["day"] = pd.to_datetime(df["day"])
    fig = go.Figure()
    fig.add_scatter(x=df.day, y=df.tx_success_rate, name="transactions (real-time)", mode="lines+markers",
                    line=dict(color=S1, width=2), marker=dict(size=5))
    fig.add_scatter(x=df.day, y=df.st_success_rate, name="settlements (final)", mode="lines+markers",
                    line=dict(color=S2, width=2), marker=dict(size=5))
    fig.update_layout(**_layout(spec["title"]))
    fig.update_yaxes(tickformat=".1%")
    return fig


def render(spec: dict, dq_days=None) -> go.Figure:
    return {"trend": lambda s: trend(s, dq_days), "contrib": contrib, "bar": bar, "sources": sources}[spec["kind"]](spec)
