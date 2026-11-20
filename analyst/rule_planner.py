"""
Deterministic planner. Used (a) when no LLM is configured, (b) as the fallback when the
model provider is unavailable or keeps returning malformed plans, and (c) as the
reference planner in evals. It covers the common question shapes; anything it can't map
is turned into a clarifying question rather than a guess.
"""
from __future__ import annotations

import copy
import re
from datetime import date, timedelta

from .plan import Plan, Window

MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"], start=1)}

DIAG_WORDS = ("why", "drop", "decline", "declin", "fell", "fall", "dip", "down", "worse",
              "contribut", "driver", "explain", "cause", "change", "changed", "deteriorat",
              "action", "recommend", "next step", "should we do", "what should we")
TREND_WORDS = ("trend", "over time", "daily", "day by day", "per day", "by day", "each day",
               "weekly", "by week", "time series")
SOURCE_WORDS = ("settlement", "finance", "reconcil", "two sources", "disagree", "mismatch")

DIM_PATTERNS = [
    ("customer_vpa", r"\b(vpa|vpas|customer|customers|payer|payers)\b"),
    ("failure_reason", r"\b(failure reason|reason|reasons|error code|decline code)s?\b"),
    ("bank_tier", r"\bbank tier\b"),
    ("bank", r"\b(bank|banks|issuer|issuers)\b"),
    ("method", r"\b(method|methods|payment method|instrument|mode|modes)\b"),
    ("merchant", r"\b(merchant|merchants)\b"),
    ("category", r"\b(category|categories|vertical|verticals|segment|segments)\b"),
    ("city_tier", r"\b(city|cities|city tier|tier)\b"),
    ("hour", r"\b(hour|hourly|time of day)\b"),
]
METHOD_ALIASES = {"upi": "UPI", "card": "Card", "cards": "Card", "credit card": "Card",
                  "debit card": "Card", "netbanking": "NetBanking", "net banking": "NetBanking",
                  "nb": "NetBanking", "wallet": "Wallet", "wallets": "Wallet"}


_TIME_UNIT = r"(?:days?|weeks?|months?|years?|hours?|mins?|minutes?)"
_NUM = r"(?:₹|rs\.?\s*|inr\s*)?\d[\d,]*(?:\.\d+)?\s*(?:k|lakh|inr|rupees|rs)?"
_CONDITION_PATTERNS = [
    (rf"\b(?:above|over|below|under|more than|less than|greater than|fewer than|at least|at most|"
     rf"exceeding|bigger than|smaller than)\s+{_NUM}(?!\s*{_TIME_UNIT}\b)", "numeric threshold"),
    (rf"(?:>=|<=|>|<)\s*{_NUM}", "numeric threshold"),
    (rf"\bbetween\s+{_NUM}\s+and\s+{_NUM}(?!\s*{_TIME_UNIT}\b)", "numeric range"),
    (r"\b(?:distinct|unique)\s+\w+", "distinct count"),
    (r"\b(?:median|percentile|p\d{2})\b", "percentile"),
    (r"\brepeat (?:customers?|payers?|users?)\b", "repeat-customer logic"),
]


def unmodelled_conditions(question: str) -> list[str]:
    """Conditions in the question that no catalog plan can apply (amount thresholds, distinct counts,
    percentiles...). A catalog plan that ignores them would answer a different question with full
    confidence, so the engine routes these to guarded free SQL or asks the user instead."""
    q = question.lower()
    found = []
    for pat, kind in _CONDITION_PATTERNS:
        for m in re.finditer(pat, q):
            txt = m.group(0).strip()
            if not re.search(r"\d|distinct|unique|median|percentile|repeat", txt):
                continue
            found.append(f"{kind}: '{txt}'")
    return found


_DOWN = r"\b(drop\w*|declin\w*|fall\w*|fell|dip\w*|down|decreas\w*|worse\w*|lower\w*|reduc\w*|deteriorat\w*)\b"
_UP = r"\b(rise|rose|risen|rising|increas\w*|up|improv\w*|grow\w*|grew|higher|jump\w*|spike\w*|went up)\b"


def premise_direction(question: str) -> str | None:
    """'down' / 'up' if the question presupposes a direction of change, else None."""
    q = question.lower()
    down, up = bool(re.search(_DOWN, q)), bool(re.search(_UP, q))
    if down and not up:
        return "down"
    if up and not down:
        return "up"
    return None


METRIC_CLARIFICATION = {
    "question": "Which measure do you mean? 'Success rate' can use two denominators.",
    "options": ["Success rate (successful / all attempts, incl. pending)",
                "Success rate excluding pending (successful / successful + failed)",
                "Failure rate", "GMV (successful amount)"]}


def metric_ambiguity(question: str) -> Plan | None:
    """Clarify plan if the question names no concrete measure but asks about 'performance', a bare
    'rate', approval/conversion etc. Shared by both planners: the engine also applies it to LLM plans,
    so "clarify rather than guess" is enforced in code, not left to the prompt."""
    q = " " + question.lower().strip() + " "
    concrete = (r"excl\w*\.? pending|without pending|terminal|success rate|\bsr\b|success ratio|success %|"
                r"failure rate|fail rate|failures|failure reason|\bgmv\b|\btpv\b|revenue|value processed|amount|"
                r"average ticket|avg ticket|\baov\b|ticket size|how many|volume|count|attempts|settlement")
    if re.search(concrete, q):
        return None
    if re.search(r"approval rate|conversion|\brate\b|performance|performing|doing", q):
        return Plan(intent="clarify", clarification=copy.deepcopy(METRIC_CLARIFICATION))
    return None


class RulePlanner:
    def __init__(self, data_start: date, data_end: date, merchants: list[str], banks: list[str],
                 categories: list[str]):
        self.data_start, self.data_end = data_start, data_end
        self.merchants, self.banks, self.categories = merchants, banks, categories

    # ----------------------------------------------------------------- windows
    def _week(self, k_back: int) -> Window:
        end = self.data_end - timedelta(days=7 * k_back)
        return Window(end - timedelta(days=6), end)

    def _prev(self, w: Window) -> Window:
        n = w.days()
        return Window(w.start - timedelta(days=n), w.start - timedelta(days=1))

    def parse_time(self, q: str) -> tuple[Window | None, Window | None, list[str]]:
        notes: list[str] = []
        ref = self.data_end
        iso = re.findall(r"(\d{4}-\d{2}-\d{2})", q)
        if len(iso) >= 2:
            w = Window(date.fromisoformat(iso[0]), date.fromisoformat(iso[1]))
            b = Window(date.fromisoformat(iso[2]), date.fromisoformat(iso[3])) if len(iso) >= 4 else None
            return w, b, notes
        if re.search(r"\b(last|past|previous) week\b|\bweek over week\b|\bwow\b|\bthis week\b", q):
            notes.append(f"'Last week' = the most recent full week in the data ({self._week(0).label()}), "
                         f"compared with the week before.")
            return self._week(0), self._week(1), notes
        m = re.search(r"\b(?:last|past) (\d+) (day|days|week|weeks)\b", q)
        if m:
            n = int(m.group(1)) * (7 if m.group(2).startswith("week") else 1)
            w = Window(ref - timedelta(days=n - 1), ref)
            notes.append(f"Compared with the preceding {n} days.")
            return w, self._prev(w), notes
        if re.search(r"\byesterday\b", q):
            w = Window(ref, ref)
            notes.append("'Yesterday' = last day in the data; compared with the same weekday a week earlier.")
            return w, Window(ref - timedelta(days=7), ref - timedelta(days=7)), notes
        if re.search(r"\b(this month|month to date|mtd)\b", q):
            w = Window(ref.replace(day=1), ref)
            prev_end = w.start - timedelta(days=1)
            b = Window(prev_end.replace(day=1), prev_end.replace(day=min(ref.day, prev_end.day)))
            notes.append("Month-to-date compared with the same days of the previous month.")
            return w, b, notes
        if re.search(r"\blast month\b", q):
            end = ref.replace(day=1) - timedelta(days=1)
            w = Window(end.replace(day=1), end)
            return w, self._prev(w), notes
        yr = re.search(r"\b(20\d{2})\b", q)
        year = int(yr.group(1)) if yr else ref.year
        found = [(MONTHS[mo], mo) for mo in MONTHS if re.search(rf"\b{mo}\b|\b{mo[:3]}\b", q)]
        if yr and not found:
            return Window(date(year, 1, 1), date(year, 12, 31)), None, notes
        if found:
            found.sort()
            def mwin(mi):
                s = date(year, mi, 1)
                e = (date(year, mi + 1, 1) if mi < 12 else date(year + 1, 1, 1)) - timedelta(days=1)
                if e < self.data_start or s > self.data_end:
                    return Window(s, e)  # outside data: engine turns this into a clarification
                return Window(max(s, self.data_start), min(e, self.data_end))
            if len(found) >= 2:
                return mwin(found[-1][0]), mwin(found[0][0]), notes
            w = mwin(found[0][0])
            b = mwin(found[0][0] - 1) if found[0][0] > 1 else None
            if b and (b.end < self.data_start or b.start > self.data_end):
                b = None
            if b and b.start == self.data_start and b.days() < 7:
                b = None
            if b:
                notes.append(f"{w.label()} compared with the previous month ({b.label()}).")
            return w, b, notes
        return None, None, notes

    # ----------------------------------------------------------------- main
    def plan(self, question: str) -> Plan:
        q = " " + question.lower().strip() + " "
        window, baseline, notes = self.parse_time(q)
        assumptions = list(notes)

        # metric
        metric = None
        if re.search(r"excl\w*\.? pending|without pending|terminal", q):
            metric = "success_rate_excl_pending"
        elif re.search(r"success rate|\bsr\b|success ratio|success %", q):
            metric = "success_rate"
        elif re.search(r"failure rate|fail rate|failures", q):
            metric = "failure_rate"
        elif re.search(r"\bgmv\b|\btpv\b|revenue|value processed|amount", q):
            metric = "gmv"
        elif re.search(r"average ticket|avg ticket|\baov\b|ticket size", q):
            metric = "avg_ticket"
        elif re.search(r"how many|volume|count|number of (transactions|payments|attempts)|attempts", q):
            metric = "attempts"
        elif metric_ambiguity(question):
            return metric_ambiguity(question)

        # intent
        if any(w in q for w in SOURCE_WORDS):
            intent = "compare_sources"
            metric = metric or "success_rate"
        elif re.search(r"\b(" + "|".join(DIAG_WORDS) + r")\w*\b", q):
            intent = "diagnose_change"
        elif any(w in q for w in TREND_WORDS):
            intent = "trend"
        else:
            intent = "breakdown"

        if metric is None:
            if intent == "diagnose_change" and re.search(r"success|fail", q):
                metric = "success_rate"
            elif intent in ("compare_sources",):
                metric = "success_rate"
            else:
                return Plan(intent="clarify", clarification={
                    "question": "Which metric should I look at?",
                    "options": ["Success rate", "Failure rate", "Attempts (volume)", "GMV"]})
            assumptions.append("Metric not named; assumed success rate.")

        # filters
        filters: dict[str, list[str]] = {}
        for alias, canon in METHOD_ALIASES.items():
            if re.search(rf"\b{re.escape(alias)}\b", q):
                filters.setdefault("method", [])
                if canon not in filters["method"]:
                    filters["method"].append(canon)
        for b in self.banks:
            short = b.lower().replace(" bank", "").replace(" federal", "").replace(" public", "").strip()
            if b.lower() in q or re.search(rf"\b{re.escape(short)}\b", q):
                filters.setdefault("bank", []).append(b)
        for mname in self.merchants:
            if re.search(rf"\b{re.escape(mname.lower())}\b", q):
                filters.setdefault("merchant", []).append(mname)
        for c in self.categories:
            if re.search(rf"\b{re.escape(c.replace('_', ' '))}\b|\b{c}\b", q):
                filters.setdefault("category", []).append(c)

        # dimensions
        dims: list[str] = []
        for key, pat in DIM_PATTERNS:
            if re.search(pat, q) and key not in dims:
                if key == "bank" and "bank_tier" in dims:
                    continue
                if key == "method" and "method" in filters and len(filters["method"]) == 1 \
                        and not re.search(r"\bby (payment )?method", q):
                    continue
                dims.append(key)
        # "UPI vs Card" style comparison -> breakdown by method
        if "method" in filters and len(filters["method"]) > 1 and "method" not in dims:
            dims.insert(0, "method")
        if "bank" in filters and len(filters["bank"]) > 1 and "bank" not in dims:
            dims.insert(0, "bank")
        # a filter on a dimension is not also a grouping unless explicitly "by"
        for k in list(filters):
            if k in dims and len(filters[k]) == 1 and not re.search(rf"\bby {k}", q):
                dims.remove(k)

        use_raw = bool(re.search(r"\braw\b|including duplicates|with duplicates", q))

        # time window handling
        if window is None:
            if intent == "diagnose_change":
                return Plan(intent="clarify", metric=metric, clarification={
                    "question": "Which period should I compare? A change needs a current and a baseline window.",
                    "options": ["Last week vs the week before", "October vs September",
                                "Last 14 days vs the 14 days before"]})
            if intent == "trend":
                window = Window(self.data_start, self.data_end)
                assumptions.append(f"No period given; showing the full dataset ({window.label()}).")
            else:
                window = Window(self.data_end - timedelta(days=6), self.data_end)
                assumptions.append(f"No period given; assumed the last 7 days in the data ({window.label()}).")
        if intent == "diagnose_change" and baseline is None:
            baseline = self._prev(window)
            assumptions.append(f"Baseline = the preceding period of equal length ({baseline.label()}).")
        if intent == "breakdown" and not dims:
            if filters:
                dims = [list(filters)[0]]
            else:
                dims = []
                intent = "breakdown"
        if intent == "breakdown" and not dims:
            # single number question ("what was the success rate last week")
            dims = []

        return Plan(intent=intent, metric=metric, dimensions=dims, filters=filters,
                    window=window, baseline=baseline if intent in ("diagnose_change", "breakdown") else None,
                    use_raw=use_raw, assumptions=assumptions)
