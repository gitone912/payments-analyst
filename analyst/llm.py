"""
LLM planner with provider abstraction, bounded retries, usage/latency/cost tracking and
graceful degradation.

The LLM's only job is to turn a question into a *plan* (JSON) that references the
semantic catalog, or - for questions the catalog can't express - a single SELECT. It
never computes numbers and never writes the narrative; numbers come from executed SQL.

Providers (stdlib only, no SDKs):
  LLM_PROVIDER=gemini  GEMINI_API_KEY=...  LLM_MODEL=gemini-2.0-flash
  LLM_PROVIDER=openai  OPENAI_API_KEY=...  LLM_MODEL=gpt-4o-mini  [OPENAI_BASE_URL=...]
      (any OpenAI-compatible endpoint: Groq, OpenRouter, Ollama, vLLM ...)
  LLM_PROVIDER=none    -> rule planner only
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import date

from .plan import Plan, Window, validate_plan
from .semantic import catalog_text

# rough USD per 1M tokens (input, output) for the cost panel; adjust to your contract
PRICES = {"gemini-2.0-flash": (0.10, 0.40), "gemini-2.5-flash": (0.30, 2.50),
          "gpt-4o-mini": (0.15, 0.60), "default": (0.50, 1.50)}


class ProviderUnavailable(Exception):
    pass


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


@dataclass
class LLMResult:
    plan: Plan | None
    calls: list[LLMCall] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


SYSTEM_PROMPT = """You are the planning component of a payments analytics assistant.
Convert the user's question into ONE JSON object (no prose, no markdown fences).

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
 "assumptions": ["..."]
}}

Rules:
- Data covers {start} to {end}. Treat {end} as "today". "Last week" = {lw_start}..{end}; its baseline = the 7 days before.
- diagnose_change = why did a metric change / what drove it / which segments contributed. Needs window AND baseline.
- If a change question gives no time period, or the metric/denominator is genuinely ambiguous, return intent "clarify" with 2-4 options. Do not guess.
- Use only metric, dimension and filter keys in the catalog. Filter values must be real names: banks {banks}; methods UPI, Card, NetBanking, Wallet; categories {cats}.
- Use "free_sql" ONLY if the catalog cannot express the question; then put one DuckDB SELECT in "sql", using only the logical tables listed (alias the fact table as t is not required).
- Record every interpretation you made in "assumptions".

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
    d = json.loads(m.group(0))
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
    )


class LLMPlanner:
    def __init__(self, data_start: date, data_end: date, banks: list[str], cats: list[str],
                 provider: str | None = None, model: str | None = None, max_attempts: int = 2):
        self.provider = (provider or os.getenv("LLM_PROVIDER", "none")).lower()
        default_model = {"gemini": "gemini-2.0-flash", "openai": "gpt-4o-mini"}.get(self.provider, "")
        self.model = model or os.getenv("LLM_MODEL", default_model)
        self.max_attempts = max_attempts
        self.data_start, self.data_end = data_start, data_end
        from datetime import timedelta
        self.system = SYSTEM_PROMPT.format(
            start=data_start, end=data_end, lw_start=data_end - timedelta(days=6),
            banks=", ".join(banks), cats=", ".join(cats), catalog=catalog_text())
        self.simulate_outage = False

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
            if e.code in (429, 500, 502, 503, 504):
                raise ProviderUnavailable(f"HTTP {e.code} from provider")
            raise RuntimeError(f"HTTP {e.code}: {e.read()[:200]!r}")
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            raise ProviderUnavailable(f"network error: {e}")

    def _complete(self, user: str) -> tuple[str, int, int]:
        if self.simulate_outage:
            raise ProviderUnavailable("simulated provider outage (demo toggle)")
        if self.provider == "gemini":
            key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if not key:
                raise ProviderUnavailable("GEMINI_API_KEY not set")
            url = (f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}"
                   f":generateContent?key={key}")
            body = {"systemInstruction": {"parts": [{"text": self.system}]},
                    "contents": [{"role": "user", "parts": [{"text": user}]}],
                    "generationConfig": {"temperature": 0, "responseMimeType": "application/json"}}
            r = self._post(url, body, {"Content-Type": "application/json"})
            text = r["candidates"][0]["content"]["parts"][0]["text"]
            u = r.get("usageMetadata", {})
            return text, u.get("promptTokenCount", 0), u.get("candidatesTokenCount", 0)
        if self.provider == "openai":
            key = os.getenv("OPENAI_API_KEY", "")
            base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
            body = {"model": self.model, "temperature": 0,
                    "response_format": {"type": "json_object"},
                    "messages": [{"role": "system", "content": self.system},
                                 {"role": "user", "content": user}]}
            r = self._post(f"{base}/chat/completions", body,
                           {"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
            u = r.get("usage", {})
            return r["choices"][0]["message"]["content"], u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
        raise ProviderUnavailable("no LLM provider configured")

    # ------------------------------------------------------------- planning
    def plan(self, question: str) -> LLMResult:
        res = LLMResult(plan=None)
        user = f"Question: {question}"
        pin, pout = PRICES.get(self.model, PRICES["default"])
        for attempt in range(1, self.max_attempts + 1):
            t0 = time.perf_counter()
            try:
                text, ti, to = self._complete(user)
            except ProviderUnavailable as e:
                ms = (time.perf_counter() - t0) * 1000
                res.calls.append(LLMCall(self.provider, self.model, attempt, ms, 0, 0, 0.0, False, str(e)))
                res.errors.append(f"provider unavailable: {e}")
                return res  # don't hammer a dead provider; caller falls back
            ms = (time.perf_counter() - t0) * 1000
            cost = ti / 1e6 * pin + to / 1e6 * pout
            try:
                plan = plan_from_json(text)
                problems = validate_plan(plan, self.data_start, self.data_end)
                if problems:
                    raise ValueError("; ".join(problems))
            except Exception as e:  # noqa: BLE001  malformed / unsupported output
                res.calls.append(LLMCall(self.provider, self.model, attempt, ms, ti, to, cost, False,
                                         f"invalid plan: {e}"))
                res.errors.append(f"attempt {attempt}: invalid plan ({e})")
                user = (f"Question: {question}\n\nYour previous output was rejected: {e}.\n"
                        f"Return a corrected JSON object only.")
                continue
            res.calls.append(LLMCall(self.provider, self.model, attempt, ms, ti, to, cost, True))
            res.plan = plan
            return res
        return res
