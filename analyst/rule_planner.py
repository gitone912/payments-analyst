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
    ("decline_type", r"\b(decline type|outcome type|technical vs\.? business|business vs\.? technical)\b"),
    ("failure_reason", r"\b(failure reason|reason|reasons|error code|decline code)s?\b"),
    ("bank_tier", r"\bbank tier\b"),
    ("bank", r"\b(bank|banks|issuer|issuers)\b"),
    ("method", r"\b(method|methods|payment method|instrument|mode|modes)\b"),
    ("merchant", r"\b(merchant|merchants)\b"),
    ("category", r"\b(category|categories|vertical|verticals|segment|segments)\b"),
    ("city_tier", r"\b(city|cities|city tier)\b|(?<!bank )\btiers?\b"),
    ("hour", r"\b(hour|hourly|time of day)\b"),
]
METHOD_ALIASES = {"upi": "UPI", "card": "Card", "cards": "Card", "credit card": "Card",
                  "debit card": "Card", "netbanking": "NetBanking", "net banking": "NetBanking",
                  "nb": "NetBanking", "wallet": "Wallet", "wallets": "Wallet"}


_MON = (r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|"
        r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)")
_DAY = r"(\d{1,2})(?:st|nd|rd|th)?"
_MON_NUM = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct",
                                         "nov", "dec"], 1)}
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
    (r"\b(?:weekends?|weekdays?|mondays?|tuesdays?|wednesdays?|thursdays?|fridays?|saturdays?|sundays?)\b",
     "day of week"),
    (r"\bbetween\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?\s+and\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?\b"
     r"|\b(?:before|after|from)\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)\b"
     r"|\b(?:mornings?|afternoons?|evenings?|nights?|overnight|business hours|office hours|peak hours|off[- ]peak)\b",
     "time of day"),
    (r"\b(?:high|low)[- ]value\b|\b(?:large|small|big|high|low)[- ](?:ticket|value|amount)s?\b"
     r"|\b(?:large|small|big) (?:payments?|transactions?|txns?)\b|\bbig[- ]ticket\b|\bmicro[- ]?payments?\b",
     "vague threshold"),
    (r"\b(?:first[- ]time|new|returning|repeat) (?:customers?|users?|payers?)\b", "customer cohort"),
]
NUMERIC_KINDS = {"numeric threshold", "numeric range"}


def unmodelled_conditions(question: str) -> list[str]:
    """Conditions in the question that no catalog plan can apply (amount thresholds, distinct counts,
    percentiles...). A catalog plan that ignores them would answer a different question with full
    confidence, so the engine routes these to guarded free SQL or asks the user instead."""
    q = question.lower()
    found = []
    for pat, kind in _CONDITION_PATTERNS:
        for m in re.finditer(pat, q):
            txt = m.group(0).strip()
            if kind in NUMERIC_KINDS and not re.search(r"\d", txt):
                continue
            item = f"{kind}: '{txt}'"
            if item not in found:
                found.append(item)
    return found


_DOWN = r"\b(drop\w*|declin\w*|fall\w*|fell|dip\w*|down|decreas\w*|worse\w*|lower\w*|reduc\w*|deteriorat\w*)\b"
_UP = r"\b(rise|rose|risen|rising|increas\w*|up|improv\w*|grow\w*|grew|higher|jump\w*|spike\w*|went up)\b"


_METRIC_NAME_PHRASES = (r"\b(?:business|customer)[- ]declines?(?: rate)?\b|\bdeclines? (?:type|rate|code)s?\b"
                        r"|\btechnical (?:failures?|declines?)(?: rate)?\b|\btechnical success(?: rate)?\b")


def strip_metric_names(q: str) -> str:
    """'business decline rate' names a metric; it does not say the metric declined."""
    return re.sub(_METRIC_NAME_PHRASES, " metric ", q.lower())


def premise_direction(question: str) -> str | None:
    """'down' / 'up' if the question presupposes a direction of change, else None."""
    q = strip_metric_names(question)
    down, up = bool(re.search(_DOWN, q)), bool(re.search(_UP, q))
    if down and not up:
        return "down"
    if up and not down:
        return "up"
    return None


_FOLLOWUP = (r"^\s*(?:and|now|then|also|ok(?:ay)?|what about|how about|same|instead)\b"
             r"|\b(?:that|it|those|them|these|same)\b|\bbreak (?:it|that|this) down\b|\bdrill (?:in|down)\b")


def is_followup(question: str) -> bool:
    """Does the question lean on the previous one ("now break that down by bank", "what about UPI")?"""
    return bool(re.search(_FOLLOWUP, question.lower()))


# Every word the deterministic planner knows how to act on (or can safely ignore). Anything else in
# a question is "unrecognised": the rule planner asks instead of answering a different question,
# and an LLM plan must say what it did with the word. Allowlist, so new phrasings fail closed.
_KNOWN_WORDS = set("""
a an the of in on at for to from by with and or vs versus between per each every all any our we us
is are was were be been being do does did doing done has have had it its this that these those them
what which who why how when where whats show give me tell list get find see compare comparison split
can could would should please about across over under than then also now ok okay just only same
instead again and so up down out into within during since until before after past previous prior
last this next current latest recent so-far far to-date till today yesterday week weeks weekly
day days daily month months monthly year years quarter quarters ytd mtd wow mom date dates period
periods time trend trends over-time series history historical moving
success successful succeeded rate rates ratio sr percent percentage share failure failures failed fail
failing decline declines declined declining drop drops dropped dropping fell fall falling fallen dip
dips dipped down worse worsen worsened lower low lowest high higher highest top bottom best worst
most least rise rose risen rising increase increased increasing improve improved improving grow
grew growth jump jumped spike spiked change changes changed changing move moved movement shift
shifted swing deteriorate deteriorated deterioration went go going gone happen happened happening
cause caused causes causing driver drivers drive drove driving drives contribute contributed
contributing contribution contributions contributor contributors explain explained explanation behind
reason reasons root impact impacted effect affected affecting
pending excluding exclude excluded excl including include included incl without with terminal
attempt attempts attempted transaction transactions txn txns payment payments volume volumes count
counts number numbers many much total totals overall gmv tpv revenue value processed amount amounts
average avg ticket size aov mean
bank banks issuer issuers tier tiers method methods mode modes instrument instruments merchant merchants
category categories vertical verticals segment segments city cities hour hours hourly breakdown
customer customers payer payers vpa vpas error code codes
upi card cards credit debit netbanking net banking nb wallet wallets
settlement settlements finance financial reconcile reconciliation reconciled agree agrees disagree
disagreement match matches mismatch consistent consistency differ differs difference source sources
two data team teams records record logs log numbers system systems report reported
raw duplicate duplicates duplicated ingested dedup deduplicated
action actions recommend recommended recommendation recommendations next step steps take should
done fix
performance performing perform doing approval conversion
technical business tsr outcome outcomes type types first second third fourth final recently lately
recent until till from today
compared comparing against relative break broke broken traffic you your not only own there their
more less very really both other others look looking need want know let like check still even
""".split())
_JANUARY_TO_DECEMBER = ("january february march april may june july august september october november "
                        "december jan feb mar apr jun jul aug sep sept oct nov dec").split()
_KNOWN_WORDS.update(_JANUARY_TO_DECEMBER)


def _plain(text: str) -> str:
    """Lowercase, punctuation inside words removed ('Co-op' -> 'coop'), single spaces."""
    return " ".join(re.sub(r"[^a-z0-9]", "", w) for w in text.lower().split())


def unrecognised_terms(question: str, extra_known: set[str] = frozenset()) -> list[str]:
    """Content words in the question that no planner rule accounts for (names of banks, merchants and
    categories are passed in as extra_known). Words inside detected conditions are left to that check."""
    q = question.lower()
    for pat, _ in _CONDITION_PATTERNS:
        q = re.sub(pat, " ", q)
    out = []
    for w in re.findall(r"[a-z][a-z'-]*", q):
        w = w.strip("'-").removesuffix("'s")
        if len(w) < 3 or w in _KNOWN_WORDS or w in extra_known or re.fullmatch(r"\d+", w):
            continue
        if w.endswith("s") and (w[:-1] in _KNOWN_WORDS or w[:-1] in extra_known):
            continue
        if w not in out:
            out.append(w)
    return out


_PART_STARTERS = (r"why|what|how|which|show|give|compare|is|are|was|were|do|does|did|can|could|plot|list|"
                  r"tell|trend|daily|weekly|monthly|gmv|tpv|success|failure|failures|attempts|avg|average|"
                  r"technical|business|top|break|drill")
_SPLIT = re.compile(rf"\s*;\s*|\s*,?\s+(?:and\s+also|also|and\s+then|then)\s+|\s*,?\s+and\s+(?=(?:{_PART_STARTERS})\b)",
                    re.IGNORECASE)
_PART_CUE = re.compile(rf"\b(?:{_PART_STARTERS}|rate|volume|ticket|settlements?)\b", re.IGNORECASE)


def split_question(question: str) -> list[str]:
    """'Show the daily trend and why did success rate drop last week' is two questions. Split where a
    new question starts ('and why...', 'and GMV...', '; ...', 'also ...'); keep the question whole when
    a piece would be too small to stand alone ('GMV and attempts by bank', 'between 9am and 5pm')."""
    raw = [p.strip(" ?.,") for p in _SPLIT.split(question) if p and p.strip(" ?.,")]
    if len(raw) < 2 or any(len(p.split()) < 3 or not _PART_CUE.search(p) for p in raw):
        return [question]
    return raw


def _typo_distance(a: str, b: str) -> int:
    """Edit distance counting a swap of two neighbouring letters as one edit ('bnak' -> 'bank')."""
    d = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        d[i][0] = i
    for j in range(len(b) + 1):
        d[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[len(a)][len(b)]


def correct_spelling(question: str, extra_known: set[str] = frozenset()) -> tuple[str, list[tuple[str, str]]]:
    """Fix obvious typos ("sucess", "faliure", "bnak", "Octobr") before planning. Only words the
    planner does not recognise are touched, and only when one known word is a close match, so a
    genuinely unknown word ("premium") is left alone and still gets a clarifying question.
    Returns the corrected question and the (wrong, right) pairs, which the answer lists."""
    vocab = (_KNOWN_WORDS | set(extra_known)) - {w for w in _KNOWN_WORDS if len(w) < 4}
    fixes: list[tuple[str, str]] = []
    out = question
    for w in unrecognised_terms(question, extra_known):
        if len(w) < 4 or not w.isalpha():
            continue
        # a typo is one slip (wrong, extra, missing or swapped letter); long words may have two.
        # A real word two edits away ("defined" vs "declined") is a different word, not a typo.
        limit = 2 if len(w) >= 9 else 1
        dists = sorted((_typo_distance(w, v), v) for v in vocab if abs(len(v) - len(w)) <= limit)
        best = [v for d, v in dists if d == dists[0][0]] if dists and dists[0][0] <= limit else []
        if len(best) > 1:  # tie: prefer the candidate with the same ending ('failurs' -> 'failures')
            same_end = [v for v in best if v[-1] == w[-1]]
            best = same_end if len(same_end) == 1 else best
        if len(best) != 1:
            continue  # nothing close, or two equally close candidates: ask instead of guessing
        right = best[0]
        out = re.sub(rf"(?i)\b{re.escape(w)}\b",
                     lambda m, r=right: r.capitalize() if m.group(0)[:1].isupper() else r, out)
        fixes.append((w, right))
    return out, fixes


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

    @staticmethod
    def _year(q: str):
        """A year like 2025, but not an amount ("over 2000 INR", "₹2025", "2000 rupees")."""
        for m in re.finditer(r"\b(20\d{2})\b", q):
            before, after = q[:m.start()][-20:], q[m.end():m.end() + 10]
            if re.search(r"(?:above|over|below|under|than|least|most|exceeding|between|and|[<>=₹]|rs\.?|inr)\s*$",
                         before) or re.match(r"\s*(?:inr|rupees|rs\b|k\b|lakh|,\d)", after):
                continue
            return m
        return None

    def _days(self, q: str, notes: list[str]):
        """Specific days and day ranges: '14 Oct', 'Oct 14th', '2026-10-14', '1 oct to 10 oct',
        '1-10 October', 'first week of October'. Returns (window, baseline) or None."""
        year = int(self._year(q).group(1)) if self._year(q) else self.data_end.year

        def mk(day, mon):
            try:
                return date(year, _MON_NUM[mon[:3]], int(day))
            except ValueError:
                return None
        m = re.search(rf"\b(first|1st|second|2nd|third|3rd|fourth|4th|last|final)\s+week\s+of\s+{_MON}\b", q)
        if m:
            mi = _MON_NUM[m.group(2)[:3]]
            month_end = (date(year, mi + 1, 1) if mi < 12 else date(year + 1, 1, 1)) - timedelta(days=1)
            k = {"first": 0, "1st": 0, "second": 1, "2nd": 1, "third": 2, "3rd": 2, "fourth": 3, "4th": 3}.get(m.group(1))
            w = (Window(date(year, mi, 1 + 7 * k), date(year, mi, 7 + 7 * k)) if k is not None
                 else Window(month_end - timedelta(days=6), month_end))
            notes.append(f"'{m.group(0)}' = {w.label()}, compared with the 7 days before.")
            return w, self._prev(w)
        pairs = [  # ranges first, so '1 oct to 10 oct' is not read as the single day '1 oct'
            (rf"\b{_DAY}\s+{_MON}\s*(?:-|–|to|till|until|and)\s*{_DAY}\s+{_MON}\b", (0, 1, 2, 3)),
            (rf"\b{_MON}\s+{_DAY}\s*(?:-|–|to|till|until|and)\s*{_MON}\s+{_DAY}\b", (1, 0, 3, 2)),
            (rf"\b{_DAY}\s*(?:-|–|to|till|until|and)\s*{_DAY}\s+{_MON}\b", (0, 2, 1, 2)),
            (rf"\b{_MON}\s+{_DAY}\s*(?:-|–|to|till|until|and)\s*{_DAY}\b", (1, 0, 2, 0)),
        ]
        for pat, (d1, m1, d2, m2) in pairs:
            m = re.search(pat, q)
            if m:
                g = m.groups()
                a, b = mk(g[d1], g[m1]), mk(g[d2], g[m2])
                if a and b and a <= b:
                    w = Window(a, b)
                    notes.append(f"{w.label()} compared with the preceding {w.days()} days.")
                    return w, self._prev(w)
        iso = re.findall(r"(\d{4}-\d{2}-\d{2})", q)
        single = None
        if len(iso) == 1:
            single = date.fromisoformat(iso[0])
        else:
            m = re.search(rf"\b{_DAY}\s+(?:of\s+)?{_MON}\b", q) or re.search(rf"\b{_MON}\s+{_DAY}\b(?!\s*,?\s*\d)", q)
            if m:
                g = m.groups()
                single = mk(g[0], g[1]) if g[0].isdigit() else mk(g[1], g[0])
        if single:
            w = Window(single, single)
            notes.append(f"{single:%d %b %Y} compared with the same weekday a week earlier.")
            return w, Window(single - timedelta(days=7), single - timedelta(days=7))
        return None

    def parse_time(self, q: str) -> tuple[Window | None, Window | None, list[str]]:
        notes: list[str] = []
        ref = self.data_end
        iso = re.findall(r"(\d{4}-\d{2}-\d{2})", q)
        if len(iso) >= 2:
            w = Window(date.fromisoformat(iso[0]), date.fromisoformat(iso[1]))
            b = Window(date.fromisoformat(iso[2]), date.fromisoformat(iso[3])) if len(iso) >= 4 else None
            return w, b, notes
        days = self._days(q, notes)
        if days:
            return days[0], days[1], notes
        if re.search(r"\btoday\b", q):
            w = Window(ref, ref)
            notes.append("'Today' = the last day in the data; compared with the same weekday a week earlier.")
            return w, Window(ref - timedelta(days=7), ref - timedelta(days=7)), notes
        if re.search(r"\b(recently|lately|recent)\b", q):
            notes.append(f"'Recently' read as the most recent full week ({self._week(0).label()}), "
                         f"compared with the week before.")
            return self._week(0), self._week(1), notes
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
            prev_end = w.start - timedelta(days=1)
            b = Window(max(prev_end.replace(day=1), self.data_start), prev_end)
            if b.days() < 7:  # a 1-day 'previous month' is not a fair comparison
                notes.append(f"{w.label()}; no comparison, because the data starts on {self.data_start:%d %b}.")
                return w, None, notes
            return w, b, notes
        yr = self._year(q)
        year = int(yr.group(1)) if yr else ref.year
        found = [(MONTHS[mo], mo) for mo in MONTHS if re.search(rf"\b{mo}\b|\b{mo[:3]}\b", q)]
        if yr and not found:
            return Window(date(year, 1, 1), date(year, 12, 31)), None, notes
        if found:
            found.sort()
            def mwin(mi, note=True):
                s = date(year, mi, 1)
                e = (date(year, mi + 1, 1) if mi < 12 else date(year + 1, 1, 1)) - timedelta(days=1)
                if e < self.data_start or s > self.data_end:
                    return Window(s, e)  # outside data: engine turns this into a clarification
                clipped = Window(max(s, self.data_start), min(e, self.data_end))
                if note and (clipped.start, clipped.end) != (s, e) and clipped.days() < 7:
                    notes.append(f"The data only covers {clipped.label()} of {s:%B %Y}.")
                return clipped
            if len(found) >= 2:
                return mwin(found[-1][0]), mwin(found[0][0], note=False), notes
            w = mwin(found[0][0])
            b = mwin(found[0][0] - 1, note=False) if found[0][0] > 1 else None
            if b and (b.end < self.data_start or b.start > self.data_end):
                b = None
            if b and b.start == self.data_start and b.days() < 7:
                b = None
            if b:
                notes.append(f"{w.label()} compared with the previous month ({b.label()}).")
            return w, b, notes
        return None, None, notes

    # ----------------------------------------------------------------- main
    def plan(self, question: str, previous: Plan | None = None) -> Plan:
        """previous: the last answered plan, used only when the question is a follow-up."""
        prev = previous if previous is not None and is_followup(question) else None
        q = " " + question.lower().strip() + " "
        # "break that down" is a breakdown request, not a "why did it go down" question
        q = re.sub(r"\bbreak(?:s|ing)? (?:it |that |this |them )?down\b", " breakdown ", q)
        window, baseline, notes = self.parse_time(q)
        assumptions = list(notes)

        # metric
        metric = None
        if re.search(r"technical success|\btsr\b", q):
            metric = "technical_success_rate"
        elif re.search(r"technical (?:failure|failures|failure rate|decline)", q):
            metric = "technical_failure_rate"
        elif re.search(r"(?:business|customer)[- ]declines?|business decline rate", q):
            metric = "business_decline_rate"
        elif re.search(r"excl\w*\.? pending|without pending|terminal", q):
            metric = "success_rate_excl_pending"
        elif re.search(r"success rate|\bsr\b|success ratio|success %", q):
            metric = "success_rate"
        elif re.search(r"failure rate|fail rate", q):
            metric = "failure_rate"
        elif re.search(r"\b(failures|failed)\b", q):
            metric = "failures"  # 'how many failed', 'most failures': a count, not a rate
        elif re.search(r"\bgmv\b|\btpv\b|revenue|value processed|amount", q):
            metric = "gmv"
        elif re.search(r"average ticket|avg ticket|\baov\b|ticket size", q):
            metric = "avg_ticket"
        elif re.search(r"how many|volume|count|number of (transactions|payments|attempts)|attempts", q):
            metric = "attempts"
        elif prev is None and metric_ambiguity(question):
            return metric_ambiguity(question)

        # intent
        if any(w in q for w in SOURCE_WORDS):
            intent = "compare_sources"
            metric = metric or "success_rate"
        elif re.search(r"\b(" + "|".join(DIAG_WORDS) + r")\w*\b", strip_metric_names(q)):
            intent = "diagnose_change"
        elif re.search(r"\b(by week|weekly|per week|each week|week by week)\b", q):
            intent = "breakdown"  # weekly values: a breakdown by week, not a daily trend
        elif any(w in q for w in TREND_WORDS):
            intent = "trend"
        else:
            intent = "breakdown"

        if prev is not None:
            inherited = []
            if metric is None:
                metric = prev.metric
                inherited.append("metric")
            if window is None and prev.window is not None:
                window, baseline = prev.window, prev.baseline
                inherited.append("period")
            if intent == "breakdown" and prev.intent in ("diagnose_change", "trend") \
                    and not re.search(r"\bbreakdown\b|\bby\b", q):
                intent = prev.intent  # "what about UPI?" re-runs the same analysis on a new slice
                inherited.append("analysis type")
            if inherited:
                assumptions.append(f"Follow-up to the previous question: kept its {', '.join(inherited)}.")

        if metric is None and re.search(r"\bfailure reasons?\b|\b(error|decline) codes?\b", q):
            metric = "failures"
            assumptions.append("Failure reasons ranked by number of failed attempts.")
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
        qn = _plain(q)  # 'Delta Coop' and 'Delta Co-op' name the same bank
        for b in self.banks:
            short = b.lower().replace(" bank", "").replace(" federal", "").replace(" public", "").strip()
            if b.lower() in q or re.search(rf"\b{re.escape(short)}\b", q) or \
                    re.search(rf"\b{re.escape(_plain(short))}\b", qn):
                filters.setdefault("bank", []).append(b)
        for mname in self.merchants:
            if re.search(rf"\b{re.escape(mname.lower())}\b", q) or re.search(rf"\b{re.escape(_plain(mname))}\b", qn):
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

        if prev is not None:
            merged = {k: list(v) for k, v in prev.filters.items() if k not in filters}
            if merged:
                assumptions.append("Kept filters from the previous question: "
                                   + ", ".join(f"{k}={'/'.join(v)}" for k, v in merged.items()) + ".")
            filters = {**merged, **filters}
        if "decline_type" in dims and metric in ("success_rate", "failure_rate", "success_rate_excl_pending"):
            metric = "attempts"  # a status-based rate is 0/100% inside each outcome type: show volumes
            assumptions.append("Split by outcome type shows attempts per type (technical failure, business "
                               "decline, pending, success).")
        if re.search(r"\b(by week|weekly|per week|each week|week by week)\b", q) and "week" not in dims:
            dims.insert(0, "week")
        use_raw = bool(re.search(r"\braw\b|including duplicates|with duplicates", q)) or bool(prev and prev.use_raw)

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
            elif "week" in dims or "day" in dims:
                window = Window(self.data_start, self.data_end)
                assumptions.append(f"No period given; showing every week in the data ({window.label()}).")
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
