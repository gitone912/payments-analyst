"""
LLM planner with provider abstraction, bounded retries, usage/latency/cost tracking and
graceful degradation.

The LLM's only job is to turn a question into a *plan* (JSON) that references the
semantic catalog, or - for questions the catalog can't express - a single SELECT. It
never computes numbers and never writes the narrative; numbers come from executed SQL.

Providers (stdlib only, no SDKs):
  LLM_PROVIDER=gemini  GEMINI_API_KEY=...  LLM_MODEL=gemini-3.5-flash-lite
  LLM_PROVIDER=openai  OPENAI_API_KEY=...  LLM_MODEL=gpt-4o-mini  [OPENAI_BASE_URL=...]
      (any OpenAI-compatible endpoint: Groq, OpenRouter, Ollama, vLLM ...)
  LLM_PROVIDER=none    -> rule planner only
"""
from __future__ import annotations

import copy
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import date

from .plan import Plan, Window, validate_plan
from .semantic import catalog_text

# rough USD per 1M tokens (input, output) for the cost panel; adjust to your contract
PRICES = {"gemini-3.5-flash-lite": (0.10, 0.40), "gemini-3.1-flash-lite": (0.10, 0.40),
          "gemini-2.5-flash": (0.30, 2.50),
          "gpt-4o-mini": (0.15, 0.60), "default": (0.50, 1.50)}


class ProviderUnavailable(Exception):
    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after  # seconds the provider asked us to wait, if it said


class CircuitBreaker:
    """Per-model breaker: after `threshold` consecutive provider failures the model is skipped for
    `cooldown_s`, so users don't each wait for a model that is down. Thread-safe (shared engine)."""

    def __init__(self, threshold: int = 3, cooldown_s: float = 60.0, clock=time.monotonic):
        self.threshold, self.cooldown_s, self.clock = threshold, cooldown_s, clock
        self._fails: dict[str, int] = {}
        self._open_until: dict[str, float] = {}
        self._lock = threading.Lock()

    def allow(self, model: str) -> bool:
        with self._lock:
            return self.clock() >= self._open_until.get(model, 0.0)

    def success(self, model: str) -> None:
        with self._lock:
            self._fails[model] = 0
            self._open_until.pop(model, None)

    def failure(self, model: str) -> None:
        with self._lock:
            self._fails[model] = self._fails.get(model, 0) + 1
            if self._fails[model] >= self.threshold:
                self._open_until[model] = self.clock() + self.cooldown_s
                self._fails[model] = 0


def _retry_after(headers, body: bytes) -> float | None:
    """Seconds to wait, from a Retry-After header or Gemini's 'retryDelay' / 'retry in Ns' text."""
    try:
        v = headers.get("Retry-After") if headers else None
        if v:
            return float(v)
    except (TypeError, ValueError):
        pass
    m = re.search(rb'"retryDelay"\s*:\s*"(\d+(?:\.\d+)?)s"|retry in (\d+(?:\.\d+)?)s', body or b"")
    return float(m.group(1) or m.group(2)) if m else None


@dataclass
class LLMCall:
    provider: str
    model: str
    attempt: int
    ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    ok: bool
    error: str | None = None
    cached: bool = False


@dataclass
class LLMResult:
    plan: Plan | None
    model: str = ""
    calls: list[LLMCall] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


SYSTEM_PROMPT = """You are the planning component of a payments analytics assistant.
Convert the user's question into ONE JSON object (no prose, no markdown fences).
You only choose WHAT to compute. The engine runs the SQL, scans segments and writes the answer.

Schema:
{{
 "intent": "diagnose_change" | "breakdown" | "trend" | "compare_sources" | "free_sql" | "clarify",
 "metric": <metric key from catalog>,
 "dimensions": [<dimension keys>],
 "filters": {{<filterable dimension>: [<exact values>]}},
 "window": {{"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}},
 "baseline": {{"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}} | null,
 "use_raw": false,
 "sql": null,
 "clarification": null | {{"question": "...", "options": ["...", "..."]}},
 "assumptions": ["..."],
 "unsupported": ["..."]
}}

Intents (pick exactly one):
- diagnose_change: WHY a metric rose/fell/changed, what drove it, which segments contributed.
  Needs window AND baseline. Leave "dimensions" empty: the engine automatically scans every
  segmentation (method, bank, category, merchant, city tier, bank x method, merchant x method).
  NEVER ask the user which dimension to diagnose by. Also use diagnose_change when the user asks
  what to do / which actions or next steps for a metric over a period: recommendations come out of
  the diagnosis, so do not clarify.
- breakdown: the value of a metric for a period, optionally split by dimensions. Use it for a
  single number ("what was the success rate in September" -> dimensions []), for "X by bank",
  for "for <merchant>" (filter + that dimension), and for comparing segments with each other
  ("UPI vs Card" -> dimensions ["method"], filters {{"method": ["UPI", "Card"]}}).
  Hour of day, day and week are catalog dimensions: "by hour of day" -> breakdown with dimensions
  ["hour"], not free_sql.
- trend: ONLY when the user asks for a daily/weekly trend or a metric "over time". dimensions [] (day is implied).
- compare_sources: ONLY for transactions vs settlements (finance) reconciliation: "do settlements
  agree", "finance numbers differ", "reconcile". Never for comparing segments or periods.
- free_sql: ONLY if no other intent can express the question, e.g. it has a condition the catalog
  cannot apply (amount thresholds like "above 5000 INR", distinct counts, percentiles). Write one
  DuckDB SELECT in "sql" over the logical tables listed, apply EVERY condition in the question
  (including the time window on created_at), and give result columns clear names
  (e.g. share_above_5000, txn_count). Express shares as a fraction between 0 and 1.
  Never select customer_vpa unless the question asks for customers.
  Day-of-week ("weekends") and time-of-day ("9am to 5pm", "evenings") conditions also need free_sql
  (e.g. dayofweek(created_at) IN (0, 6), hour(created_at) BETWEEN 9 AND 16), still inside the period asked.
  For vague terms with no standard definition ("high-value", "big transactions", "peak hours"), pick a
  concrete definition and state it in "assumptions" (e.g. "high-value = amount_inr > 10000").
- clarify: ONLY when (a) a change/why question gives no time period at all, or (b) the measure is
  genuinely unclear ("how are we performing", "how are we doing"). Give 2-4 concrete options.
  A missing dimension is NOT a reason to clarify.

Time rules:
- Data covers {start} to {end}. Treat {end} as "today".
- "last week" = {lw_start}..{end}; its baseline = {pw_start}..{pw_end}. For any "last week"
  question (breakdown too) include that baseline.
- A calendar month = the whole month, clipped to the data end for the current month
  (the latest month -> {cur_month_start}..{end}). Baseline for a month = the previous calendar month.
- If the user names a period outside the data, still return exactly that period (with the year
  they said); the engine will tell them it has no data. Do not clarify and do not move the period.
- "raw", "as ingested", "including duplicates" -> use_raw true. Otherwise false.

Other rules:
- Use only metric, dimension and filter keys in the catalog. Filter values must be real names:
  banks {banks}; methods UPI, Card, NetBanking, Wallet; categories {cats}; merchants by name.
- "failures"/"most failures" -> metric failure_rate. Customer / VPA questions -> dimension customer_vpa
  (access control is handled by the engine; just plan it).
- Record every interpretation you made in "assumptions".
- Follow-ups: if a previous question and plan are given and the new question refers to them ("that",
  "it", "what about UPI", "now by bank"), keep the previous metric, window, baseline, filters and intent
  unless the new question changes them. "Break that down by X" -> breakdown by X over the same period.
- "unsupported": list every condition in the question that your plan does NOT apply. A catalog intent
  with a non-empty "unsupported" list is wrong: use free_sql (or clarify) instead. Normally [].

Examples:
Q: What drove the rise in failure rate last week?
{{"intent":"diagnose_change","metric":"failure_rate","dimensions":[],"filters":{{}},"window":{{"start":"{lw_start}","end":"{end}"}},"baseline":{{"start":"{pw_start}","end":"{pw_end}"}},"use_raw":false,"sql":null,"clarification":null,"assumptions":["last week vs the week before"]}}
Q: What was GMV last month?
{{"intent":"breakdown","metric":"gmv","dimensions":[],"filters":{{}},"window":{{"start":"{prev_month_start}","end":"{prev_month_end}"}},"baseline":null,"use_raw":false,"sql":null,"clarification":null,"assumptions":[]}}
Q: Success rate for ShopKart by method last week
{{"intent":"breakdown","metric":"success_rate","dimensions":["method"],"filters":{{"merchant":["ShopKart"]}},"window":{{"start":"{lw_start}","end":"{end}"}},"baseline":{{"start":"{pw_start}","end":"{pw_end}"}},"use_raw":false,"sql":null,"clarification":null,"assumptions":[]}}
Q: Do finance settlement numbers match transactions this month?
{{"intent":"compare_sources","metric":"success_rate","dimensions":[],"filters":{{}},"window":{{"start":"{cur_month_start}","end":"{end}"}},"baseline":null,"use_raw":false,"sql":null,"clarification":null,"assumptions":[]}}
Q: Why did failure rate go up?
{{"intent":"clarify","metric":"failure_rate","dimensions":[],"filters":{{}},"window":null,"baseline":null,"use_raw":false,"sql":null,"clarification":{{"question":"Which period should I compare?","options":["Last week vs the week before","October vs September","Last 14 days vs the 14 days before"]}},"assumptions":[]}}

CATALOG
{catalog}
"""


def _parse_window(d) -> Window | None:
    if not d:
        return None
    return Window(date.fromisoformat(d["start"]), date.fromisoformat(d["end"]))


def plan_from_json(text: str) -> Plan:
    t = text.strip()
    t = re.sub(r"^```(?:json)?|```$", "", t, flags=re.M).strip()
    m = re.search(r"\{.*\}", t, flags=re.S)
    if not m:
        raise ValueError("no JSON object in model output")
    raw = m.group(0)
    try:
        d = json.loads(raw)
    except json.JSONDecodeError:
        # tolerate the commonest model slip (trailing commas); anything else goes back to the model
        d = json.loads(re.sub(r",\s*([}\]])", r"\1", raw))
    return Plan(
        intent=d.get("intent", ""),
        metric=d.get("metric") or "success_rate",
        dimensions=list(d.get("dimensions") or []),
        filters={k: list(v) if isinstance(v, list) else [v] for k, v in (d.get("filters") or {}).items()},
        window=_parse_window(d.get("window")),
        baseline=_parse_window(d.get("baseline")),
        use_raw=bool(d.get("use_raw", False)),
        sql=d.get("sql"),
        clarification=d.get("clarification"),
        assumptions=list(d.get("assumptions") or []),
        unsupported=[str(x) for x in (d.get("unsupported") or [])],
    )


class LLMPlanner:
    def __init__(self, data_start: date, data_end: date, banks: list[str], cats: list[str],
                 provider: str | None = None, model: str | None = None, max_attempts: int = 2):
        self.provider = (provider or os.getenv("LLM_PROVIDER", "none")).lower()
        default_model = {"gemini": "gemini-3.5-flash-lite", "openai": "gpt-4o-mini"}.get(self.provider, "")
        self.model = model or os.getenv("LLM_MODEL", default_model)
        # Backup models, tried in order when the primary is overloaded or rate-limited
        default_fallbacks = {"gemini": "gemini-3.1-flash-lite"}.get(self.provider, "")
        fallbacks = os.getenv("LLM_FALLBACK_MODELS", default_fallbacks)
        self.models = [self.model] + [m.strip() for m in fallbacks.split(",") if m.strip() and m.strip() != self.model]
        self.breaker = CircuitBreaker(int(os.getenv("LLM_BREAKER_THRESHOLD", "3")),
                                      float(os.getenv("LLM_BREAKER_COOLDOWN_S", "60")))
        self.max_retry_wait_s = float(os.getenv("LLM_MAX_RETRY_WAIT_S", "5"))   # honour short Retry-After only
        self.request_budget_s = float(os.getenv("LLM_REQUEST_BUDGET_S", "20"))  # whole request, all models
        self._sleep = time.sleep
        self.max_attempts = max_attempts
        self.data_start, self.data_end = data_start, data_end
        from datetime import timedelta
        lw_start = data_end - timedelta(days=6)
        month_start = data_end.replace(day=1)
        prev_end = month_start - timedelta(days=1)
        self.system = SYSTEM_PROMPT.format(
            start=data_start, end=data_end, lw_start=lw_start,
            pw_start=lw_start - timedelta(days=7), pw_end=lw_start - timedelta(days=1),
            cur_month_start=month_start, prev_month_start=prev_end.replace(day=1), prev_month_end=prev_end,
            banks=", ".join(banks), cats=", ".join(cats), catalog=catalog_text())
        self.simulate_outage = False
        # successful plans keyed by normalised question: re-asks (UI reruns, evals) cost no quota
        self._cache: dict[str, Plan] = {}
        # optional client-side pacing for rate-limited tiers (e.g. Gemini free tier = 15 req/min)
        self.min_interval_s = float(os.getenv("LLM_MIN_INTERVAL_S", "0"))
        self._last_call = 0.0
        self._pace_lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.provider in ("gemini", "openai")

    # ------------------------------------------------------------- transport
    def _post(self, url: str, body: dict, headers: dict, timeout: float = 20) -> dict:
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            body = e.read() or b""
            if e.code in (429, 500, 502, 503, 504):
                raise ProviderUnavailable(f"HTTP {e.code} from provider", _retry_after(e.headers, body))
            if e.code == 404:  # retired / unknown model: treat as unavailable so the next model is tried
                raise ProviderUnavailable(f"HTTP 404 (model not available) from provider")
            raise RuntimeError(f"HTTP {e.code}: {body[:200]!r}")
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            raise ProviderUnavailable(f"network error: {e}")

    def _complete(self, user: str, simulate_outage: bool = False, model: str | None = None) -> tuple[str, int, int]:
        model = model or self.model
        if simulate_outage or self.simulate_outage:
            raise ProviderUnavailable("simulated provider outage (demo toggle)")
        if self.min_interval_s > 0:
            with self._pace_lock:  # reserve a slot so concurrent requests stay under the rate limit
                slot = max(time.monotonic(), self._last_call + self.min_interval_s)
                self._last_call = slot
            if slot > time.monotonic():
                time.sleep(slot - time.monotonic())
        if self.provider == "gemini":
            key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if not key:
                raise ProviderUnavailable("GEMINI_API_KEY not set")
            # key goes in a header, not the URL, so it never lands in logs or tracebacks
            url = (f"https://generativelanguage.googleapis.com/v1beta/models/{model}"
                   f":generateContent")
            body = {"systemInstruction": {"parts": [{"text": self.system}]},
                    "contents": [{"role": "user", "parts": [{"text": user}]}],
                    "generationConfig": {"temperature": 0, "responseMimeType": "application/json"}}
            r = self._post(url, body, {"Content-Type": "application/json", "x-goog-api-key": key})
            text = r["candidates"][0]["content"]["parts"][0]["text"]
            u = r.get("usageMetadata", {})
            return text, u.get("promptTokenCount", 0), u.get("candidatesTokenCount", 0)
        if self.provider == "openai":
            key = os.getenv("OPENAI_API_KEY", "")
            base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
            body = {"model": model, "temperature": 0,
                    "response_format": {"type": "json_object"},
                    "messages": [{"role": "system", "content": self.system},
                                 {"role": "user", "content": user}]}
            r = self._post(f"{base}/chat/completions", body,
                           {"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
            u = r.get("usage", {})
            return r["choices"][0]["message"]["content"], u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
        raise ProviderUnavailable("no LLM provider configured")

    def _resilient_complete(self, user: str, outage: bool, res: "LLMResult", attempt: int):
        """One logical completion: primary model, then backups. Each model gets at most two calls
        (the second only after a short provider-requested wait), models whose breaker is open are
        skipped, and the whole thing stays inside request_budget_s. Raises ProviderUnavailable when
        every model failed, so the engine can fall back to the deterministic planner."""
        if outage or self.simulate_outage:
            res.calls.append(LLMCall(self.provider, self.model, attempt, 0.0, 0, 0, 0.0, False,
                                     "simulated provider outage (demo toggle)"))
            raise ProviderUnavailable("simulated provider outage (demo toggle)")
        deadline = time.monotonic() + self.request_budget_s
        errors = []
        for model in self.models:
            if not self.breaker.allow(model):
                errors.append(f"{model}: skipped (circuit open after repeated failures)")
                continue
            for tryno in (1, 2):
                if time.monotonic() >= deadline:
                    errors.append("request time budget exhausted")
                    raise ProviderUnavailable("; ".join(errors))
                t0 = time.perf_counter()
                try:
                    text, ti, to = self._complete(user, False, model)
                except ProviderUnavailable as e:
                    ms = (time.perf_counter() - t0) * 1000
                    self.breaker.failure(model)
                    res.calls.append(LLMCall(self.provider, model, attempt, ms, 0, 0, 0.0, False, str(e)))
                    errors.append(f"{model}: {e}")
                    wait = e.retry_after
                    if (tryno == 1 and wait is not None and wait <= self.max_retry_wait_s
                            and time.monotonic() + wait < deadline and self.breaker.allow(model)):
                        self._sleep(wait)  # provider told us how long; it's short, so wait and retry
                        continue
                    break  # no usable wait hint: move on to the next model
                self.breaker.success(model)
                return text, ti, to, model, (time.perf_counter() - t0) * 1000
        raise ProviderUnavailable("; ".join(errors) or "no model available")

    # ------------------------------------------------------------- planning
    def plan(self, question: str, feedback: str | None = None, context: str | None = None,
             simulate_outage: bool = False) -> LLMResult:
        res = LLMResult(plan=None)
        ck = (" ".join(question.lower().split()) + (f" || {feedback}" if feedback else "")
              + (f" || {context}" if context else ""))
        outage = simulate_outage or self.simulate_outage
        if ck in self._cache and not outage:
            res.plan = copy.deepcopy(self._cache[ck])
            res.calls.append(LLMCall(self.provider, self.model, 0, 0.0, 0, 0, 0.0, True, cached=True))
            res.model = self.model
            return res
        user = ((f"{context}\n\n" if context else "") + f"Question: {question}"
                + (f"\n\n{feedback}" if feedback else ""))
        for attempt in range(1, self.max_attempts + 1):
            try:
                text, ti, to, model, ms = self._resilient_complete(user, outage, res, attempt)
            except ProviderUnavailable as e:
                res.errors.append(f"provider unavailable: {e}")
                return res  # every model failed within the budget; caller falls back to rules
            pin, pout = PRICES.get(model, PRICES["default"])
            cost = ti / 1e6 * pin + to / 1e6 * pout
            try:
                plan = plan_from_json(text)
                problems = validate_plan(plan, self.data_start, self.data_end)
                if problems:
                    raise ValueError("; ".join(problems))
            except Exception as e:  # noqa: BLE001  malformed / unsupported output
                res.calls.append(LLMCall(self.provider, model, attempt, ms, ti, to, cost, False,
                                         f"invalid plan: {e}"))
                res.errors.append(f"attempt {attempt}: invalid plan ({e})")
                user = ((f"{context}\n\n" if context else "") + f"Question: {question}\n\n"
                        f"Your previous output was rejected: {e}.\nReturn a corrected JSON object only.")
                continue
            res.calls.append(LLMCall(self.provider, model, attempt, ms, ti, to, cost, True))
            res.plan = plan
            res.model = model
            self._cache[ck] = copy.deepcopy(plan)
            return res
        return res
