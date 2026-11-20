"""The analysis plan: the contract between the planner (LLM or rules) and the engine."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date

from .semantic import DIMENSIONS, FILTERABLE, METRICS

INTENTS = ("diagnose_change", "breakdown", "trend", "compare_sources", "free_sql", "clarify")


@dataclass
class Window:
    start: date
    end: date  # inclusive

    def label(self) -> str:
        return f"{self.start:%d %b} - {self.end:%d %b %Y}"

    def days(self) -> int:
        return (self.end - self.start).days + 1


@dataclass
class Plan:
    intent: str
    metric: str = "success_rate"
    dimensions: list[str] = field(default_factory=list)
    filters: dict[str, list[str]] = field(default_factory=dict)
    window: Window | None = None
    baseline: Window | None = None
    use_raw: bool = False
    sql: str | None = None                      # free_sql only
    clarification: dict | None = None           # {"question": str, "options": [str]}
    assumptions: list[str] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)  # conditions the catalog plan cannot apply

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("window", "baseline"):
            if d[k]:
                d[k] = {"start": str(d[k]["start"]), "end": str(d[k]["end"])}
        return d


class PlanValidationError(Exception):
    pass


def validate_plan(p: Plan, data_start: date, data_end: date) -> list[str]:
    """Returns a list of problems; empty = valid. Used to reject malformed LLM output."""
    errs = []
    if p.intent not in INTENTS:
        errs.append(f"unknown intent '{p.intent}'")
    if p.intent == "clarify":
        if not p.clarification or not p.clarification.get("question"):
            errs.append("clarify intent needs clarification.question")
        return errs
    if p.intent == "free_sql":
        if not p.sql or not p.sql.strip():
            errs.append("free_sql intent needs sql")
        return errs
    if p.metric not in METRICS:
        errs.append(f"unknown metric '{p.metric}'")
    for d in p.dimensions:
        if d not in DIMENSIONS:
            errs.append(f"unknown dimension '{d}'")
    for f in p.filters:
        if f not in FILTERABLE:
            errs.append(f"unknown filter dimension '{f}'")
    if p.window is None:
        errs.append("missing time window")
    for nm, w in (("window", p.window), ("baseline", p.baseline)):
        if w is None:
            continue
        if w.start > w.end:
            errs.append(f"{nm} start after end")
    if p.intent == "diagnose_change":
        if p.baseline is None:
            errs.append("diagnose_change needs a baseline window")
    return errs
