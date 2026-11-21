"""
Orchestrator: question -> plan -> validate -> guard -> execute -> analyse -> answer.

Every number in an answer comes from an executed, logged query; every fact carries the
query ids that support it. The LLM (if configured) only produces the plan.
"""
from __future__ import annotations

import copy
import json
import numbers
import re
import time
from dataclasses import dataclass, field
from datetime import date, timedelta

import pandas as pd
import sqlglot
from sqlglot import exp

from . import compiler
from .dq import DQFinding, run_checks
from .drivers import Decomposition, decompose, overall_change_ci
from .executor import Executor, QueryRecord, QueryTimeout
from .guard import SqlGuard, UnsafeQuery
from .llm import LLMCall, LLMPlanner, plan_from_json
from .plan import Plan, Window, validate_plan
from .rbac import ROLES, AccessDenied, Role
from .rule_planner import (RulePlanner, is_followup, metric_ambiguity, premise_direction,
                           unmodelled_conditions, unrecognised_terms, correct_spelling)
from .semantic import DIMENSIONS, METRICS

COMBO_DIMS = [("bank", "method"), ("merchant", "method")]


def fmt(metric: str, v: float) -> str:
    f = METRICS[metric].fmt
    if v is None or pd.isna(v):
        return "n/a"
    if f == "pct":
        return f"{v:.2%}"
    if f == "inr":
        return f"₹{v:,.0f}"
    return f"{v:,.0f}"


def pts(v: float) -> str:
    return f"{v * 100:+.2f} pts"


@dataclass
class Step:
    name: str
    status: str
    detail: str
    ms: float = 0.0


@dataclass
class Answer:
    question: str
    role: str
    status: str = "answered"            # answered | clarify | denied | blocked | error
    message: str = ""
    plan: dict | None = None
    planner_used: str = ""
    steps: list[Step] = field(default_factory=list)
    queries: list[QueryRecord] = field(default_factory=list)
    llm_calls: list[LLMCall] = field(default_factory=list)
    facts: list[dict] = field(default_factory=list)
    hypotheses: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    citation: dict = field(default_factory=dict)
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    charts: list[dict] = field(default_factory=list)
    confidence: dict = field(default_factory=dict)
    clarification: dict | None = None
    denial_rule: str | None = None
    total_ms: float = 0.0
    headline: str = ""
    user: str = ""
    # per-request planning context. Lives on the Answer, never on the shared engine: one engine
    # instance serves every user, so request state stored on it would leak between users.
    prev_plan: Plan | None = field(default=None, repr=False, compare=False)
    llm_context: str | None = field(default=None, repr=False, compare=False)
    simulate_outage: bool = field(default=False, repr=False, compare=False)
    unaddressed_terms: list[str] = field(default_factory=list, repr=False, compare=False)


class AnalystEngine:
    def __init__(self, db_path: str, llm_provider: str | None = None, audit_path: str | None = None):
        self.ex = Executor(db_path, audit_path=audit_path)
        m = self.ex.raw("SELECT merchant_id, merchant_name, category, kam_portfolio FROM raw.merchants")
        self.merchants = m
        lookup = {}
        for r in m.itertuples():
            d = {"id": r.merchant_id, "name": r.merchant_name, "portfolio": r.kam_portfolio}
            lookup[r.merchant_id.lower()] = d
            lookup[r.merchant_name.lower()] = d
        self.guard = SqlGuard(lookup)
        rng = self.ex.raw("SELECT MIN(created_at)::DATE AS s, MAX(created_at)::DATE AS e FROM raw.transactions")
        self.data_start: date = rng.s[0].date() if hasattr(rng.s[0], "date") else rng.s[0]
        self.data_end: date = rng.e[0].date() if hasattr(rng.e[0], "date") else rng.e[0]
        banks = self.ex.raw("SELECT bank_name FROM raw.banks ORDER BY 1").bank_name.tolist()
        cats = sorted(m.category.unique().tolist())
        self.rules = RulePlanner(self.data_start, self.data_end, m.merchant_name.tolist(), banks, cats)
        self.known_names: set[str] = set()
        tiers = self.ex.raw("SELECT DISTINCT city_tier FROM raw.merchants").city_tier.tolist()
        for n in [*m.merchant_name, *cats, *banks, *tiers]:
            self.known_names.update(re.findall(r"[a-z][a-z0-9]*", str(n).lower().replace("_", " ")))
        self.llm = LLMPlanner(self.data_start, self.data_end, banks, cats, provider=llm_provider)
        self.dq: list[DQFinding] = run_checks(self.ex)

    # ------------------------------------------------------------ helpers
    def visible_merchants(self, role: Role) -> set[str] | None:
        if role.merchant_scope_sql is None:
            return None
        df = self.ex.raw(f"SELECT merchant_id FROM raw.merchants WHERE {role.merchant_scope_sql}")
        return set(df.merchant_id)

    def _run(self, ans: Answer, role: Role, sql: str, purpose: str) -> tuple[pd.DataFrame, QueryRecord]:
        g = self.guard.check(sql, role, self.visible_merchants(role))
        df, rec = self.ex.run(g.executable_sql, original_sql=g.original_sql, purpose=purpose,
                              role=role.key, checks=g.checks, query_id=f"Q{len(ans.queries) + 1}",
                              user=ans.user)
        ans.queries.append(rec)
        return df, rec

    # ------------------------------------------------------------ planning
    def make_plan(self, question: str, ans: Answer, mode: str, previous: dict | None = None) -> Plan:
        if previous and previous.get("plan") and is_followup(question):
            ans.prev_plan = plan_from_json(json.dumps(previous["plan"]))
            ans.llm_context = (f"Previous question: {previous.get('question', '')}\n"
                               f"Previous plan: {json.dumps(previous['plan'], default=str)}")
            ans.steps.append(Step("follow_up", "ok", f"treated as a follow-up to: {previous.get('question', '')}"))
        plan = self._make_plan(question, ans, mode)
        plan = self._coverage_guard(question, plan, ans, mode)
        if ans.planner_used.startswith("LLM") and plan.intent != "clarify":
            plan = self._period_guard(question, plan, ans, mode)
        return plan

    def _period_problems(self, question: str, plan: Plan, prev_plan: Plan | None = None) -> list[str]:
        """Does the plan stay inside the period the question names? Checked deterministically, because a
        model that answers 'weekends last week' with 18-19 Oct (Sun + Mon) is wrong but looks right."""
        rw, _, _ = self.rules.parse_time(" " + question.lower().strip() + " ")
        if rw is None and prev_plan is not None:
            rw = prev_plan.window
        if rw is None:
            return []
        lo, hi = rw.start, rw.end
        if plan.intent == "free_sql":
            dates = [date.fromisoformat(d) for d in re.findall(r"\b(\d{4}-\d{2}-\d{2})\b", plan.sql or "")]
            if not dates:
                return [f"the SQL has no date filter, but the question asks about {rw.label()}"]
            bad = [d for d in dates if not (lo <= d <= hi + timedelta(days=1))]
            return [f"the SQL uses {', '.join(map(str, bad))}, outside {lo}..{hi}"] if bad else []
        if plan.window and (plan.window.start < lo or plan.window.end > hi):
            return [f"the plan uses {plan.window.start}..{plan.window.end}, outside {lo}..{hi}"]
        return []

    @staticmethod
    def _condition_problems(question: str, plan: Plan) -> list[str]:
        """Model-written SQL must visibly apply each condition the question states."""
        if plan.intent != "free_sql" or not plan.sql:
            return []
        sql = plan.sql.lower()
        sql_digits = set(re.findall(r"\d+(?:\.\d+)?", sql.replace(",", "")))
        probs = []
        for g in unmodelled_conditions(question):
            kind, txt = g.split(": ", 1)
            txt = txt.strip("'")
            if kind in ("numeric threshold", "numeric range"):
                for num, unit in re.findall(r"(\d[\d,]*(?:\.\d+)?)\s*(k|lakh)?", txt):
                    v = float(num.replace(",", "")) * {"k": 1_000, "lakh": 100_000}.get(unit, 1)
                    want = {f"{v:g}", str(int(v)) if v == int(v) else f"{v:g}"}
                    if not want & sql_digits:
                        probs.append(f"the SQL does not use the threshold {txt}")
            elif kind == "day of week" and not re.search(r"dayofweek|\bdow\b|isodow|dayname|weekday|strftime", sql):
                probs.append(f"the SQL does not filter on day of week ('{txt}')")
            elif kind == "time of day" and not re.search(r"\bhour\b|strftime|extract|date_part", sql):
                probs.append(f"the SQL does not filter on time of day ('{txt}')")
            elif kind == "vague threshold" and "amount_inr" not in sql:
                probs.append(f"the SQL does not define '{txt}' on amount_inr")
        return probs

    def _unaddressed(self, question: str, plan: Plan) -> list[str]:
        """Words outside the planner vocabulary that the LLM plan neither applies nor explains."""
        text = json.dumps(plan.to_dict(), default=str).lower()
        return [w for w in unrecognised_terms(question, self.known_names) if w not in text]

    def _period_guard(self, question: str, plan: Plan, ans: Answer, mode: str) -> Plan:
        probs = (self._period_problems(question, plan, ans.prev_plan)
                 + self._condition_problems(question, plan))
        if not probs:
            return self._terms_check(question, plan, ans)
        ans.steps.append(Step("period_check", "fail", "; ".join(probs)))
        res = self.llm.plan(question, feedback=(
            "Your plan failed validation: " + "; ".join(probs)
            + ". Every date must lie inside the period the question asks about, and the SQL must apply every "
              "condition in the question. Return a corrected plan."),
            context=ans.llm_context, simulate_outage=ans.simulate_outage)
        ans.llm_calls.extend(res.calls)
        if res.plan is not None and not (self._period_problems(question, res.plan, ans.prev_plan)
                                         + self._condition_problems(question, res.plan)):
            ans.steps.append(Step("period_check", "fixed", "model corrected its plan after feedback"))
            return self._terms_check(question, self._coverage_guard(question, res.plan, ans, mode), ans)
        ans.steps.append(Step("period_check", "fallback", "model could not produce a valid plan; using the "
                                                          "deterministic planner"))
        ans.caveats.append("The language model's plan failed validation (dates or conditions), so the "
                           "deterministic planner was used.")
        ans.planner_used = "Deterministic rule planner (LLM plan rejected by period check)"
        return self._coverage_guard(question, self.rules.plan(question, ans.prev_plan), ans, "rules")

    def _coverage_guard(self, question: str, plan: Plan, ans: Answer, mode: str) -> Plan:
        """A catalog plan that silently drops a condition ("above 5000 INR") answers a different
        question with full confidence. Detect that and route to guarded LLM SQL, or ask."""
        if plan.intent == "free_sql":
            return plan
        gaps = list(dict.fromkeys(unmodelled_conditions(question) + list(plan.unsupported)))
        if not ans.planner_used.startswith("LLM"):
            # the rule planner only understands words it has rules for: anything else means it may be
            # answering a different question, so it must not answer (fail closed)
            gaps += [f"unrecognised term: '{w}'" for w in unrecognised_terms(question, self.known_names)]
        if not gaps:
            return plan
        if plan.intent == "clarify" and ans.planner_used.startswith("LLM"):
            return plan  # the model already chose to ask
        ans.steps.append(Step("coverage_guard", "warn",
                              f"plan '{plan.intent}' cannot apply: {'; '.join(gaps)}"))
        if mode in ("auto", "llm") and self.llm.enabled and not (ans.simulate_outage or self.llm.simulate_outage):
            res = self.llm.plan(question, feedback=(
                "Your catalog plan cannot apply these conditions from the question: " + "; ".join(gaps)
                + ". Return intent free_sql with ONE SELECT that applies every condition, or clarify."),
                context=ans.llm_context, simulate_outage=ans.simulate_outage)
            ans.llm_calls.extend(res.calls)
            if res.plan is not None and res.plan.intent in ("free_sql", "clarify"):
                ans.planner_used = f"LLM ({self.llm.provider}:{res.model or self.llm.model}) - free SQL"
                ans.steps.append(Step("coverage_guard", "rerouted",
                                      f"re-planned as {res.plan.intent} so every condition is applied"))
                return res.plan
        ans.steps.append(Step("coverage_guard", "clarify", "no plan applies every condition; asking"))
        cond = "; ".join(g.split(": ", 1)[-1] for g in gaps)
        unknown = all(g.startswith("unrecognised term") for g in gaps)
        return Plan(intent="clarify", metric=plan.metric, clarification={
            "question": (f"I don't know how to apply {cond} to this data, and I won't answer while ignoring it. "
                         "Rephrase using metrics, banks, methods, merchants, categories or dates, or enable "
                         "the LLM planner." if unknown else
                         f"I can't apply {cond} with the standard metrics, and I won't answer while ignoring it. "
                         "Enable the LLM planner (it can write a guarded custom query), use the SQL playground, "
                         "or drop the condition."),
            "options": []})

    def _terms_check(self, question: str, plan: Plan, ans: Answer) -> Plan:
        if plan.intent == "clarify":
            return plan
        missing = self._unaddressed(question, plan)
        if not missing:
            return plan
        ans.steps.append(Step("terms_check", "warn", f"plan does not mention: {', '.join(missing)}"))
        res = self.llm.plan(question, feedback=(
            "Your plan does not show how it applies these words from the question: " + ", ".join(missing)
            + ". Apply them (free_sql if needed), or list them in 'unsupported' / explain in 'assumptions' "
              "why they do not change the query."), context=ans.llm_context, simulate_outage=ans.simulate_outage)
        ans.llm_calls.extend(res.calls)
        if (res.plan is not None and not self._unaddressed(question, res.plan)
                and not self._period_problems(question, res.plan, ans.prev_plan)
                and not self._condition_problems(question, res.plan)):
            ans.steps.append(Step("terms_check", "fixed", "model accounted for every term"))
            if res.plan.unsupported and res.plan.intent not in ("free_sql", "clarify"):
                return self._coverage_guard(question, res.plan, ans, "llm")
            return res.plan
        ans.unaddressed_terms = missing
        return plan

    def _make_plan(self, question: str, ans: Answer, mode: str) -> Plan:
        t0 = time.perf_counter()
        if mode in ("auto", "llm") and self.llm.enabled:
            res = self.llm.plan(question, context=ans.llm_context, simulate_outage=ans.simulate_outage)
            ans.llm_calls.extend(res.calls)
            if res.plan is not None:
                ans.planner_used = f"LLM ({self.llm.provider}:{res.model or self.llm.model})"
                if res.model and res.model != self.llm.model:
                    ans.steps.append(Step("llm_fallback", "ok", f"primary model {self.llm.model} unavailable; "
                                          f"answered by backup model {res.model}"))
                ans.steps.append(Step("plan", "ok", f"{ans.planner_used}, {len(res.calls)} call(s)",
                                      (time.perf_counter() - t0) * 1000))
                amb = metric_ambiguity(question) if ans.prev_plan is None else None
                if amb is not None and res.plan.intent != "clarify":
                    ans.steps.append(Step("ambiguity_guard", "override",
                                          f"LLM chose metric '{res.plan.metric}' but the question names no "
                                          "measure; asking instead of guessing"))
                    return amb
                return res.plan
            reason = "; ".join(res.errors) or "no plan"
            if mode == "llm":
                raise RuntimeError(f"LLM planner failed: {reason}")
            ans.steps.append(Step("plan", "fallback", f"LLM planner failed ({reason}); "
                                  "fell back to deterministic planner"))
            ans.caveats.append("The language model was unavailable or returned an invalid plan, so the "
                               "deterministic planner was used. Supported question types still work.")
        plan = self.rules.plan(question, ans.prev_plan)
        ans.planner_used = ans.planner_used or "Deterministic rule planner"
        ans.steps.append(Step("plan", "ok", "Deterministic rule planner", (time.perf_counter() - t0) * 1000))
        return plan

    def _clip(self, w: Window | None, ans: Answer) -> Window | None:
        if w is None:
            return None
        if w.end < self.data_start or w.start > self.data_end:
            return None
        s, e = max(w.start, self.data_start), min(w.end, self.data_end)
        if (s, e) != (w.start, w.end):
            ans.assumptions.append(f"Requested period clipped to available data ({s:%d %b} - {e:%d %b %Y}).")
        return Window(s, e)

    # ------------------------------------------------------------ entry
    def ask(self, question: str, role_key: str = "head_of_payments", planner_mode: str = "auto",
            previous: dict | None = None, *, user: str = "", simulate_outage: bool = False) -> Answer:
        """previous: {"question": ..., "plan": ...} of the last answered question, for follow-ups.
        user: authenticated identity, written to the audit log. All per-request state lives on the
        returned Answer, so one engine can serve concurrent users safely."""
        t0 = time.perf_counter()
        role = ROLES[role_key]
        ans = Answer(question=question, role=role.label, user=user, simulate_outage=simulate_outage)
        try:
            corrected, fixes = correct_spelling(question, self.known_names)
            if fixes:  # plan on the corrected text, and say exactly what was changed
                note = "Read " + ", ".join(f"'{a}' as '{b}'" for a, b in fixes) + "."
                ans.question = question = corrected
                ans.steps.append(Step("spelling", "ok", note))
                ans.assumptions.append(note)
            plan = self.make_plan(question, ans, planner_mode, previous)
            ans.plan = plan.to_dict()
            ans.assumptions.extend(plan.assumptions)

            problems = validate_plan(plan, self.data_start, self.data_end)
            if plan.intent == "clarify" or (problems and plan.intent != "clarify"):
                ans.status = "clarify"
                ans.clarification = plan.clarification or {
                    "question": "I couldn't map that question to the data. Could you rephrase?",
                    "options": ["Why did success rate drop last week?", "Success rate by bank last week",
                                "Daily success rate trend", "Do settlements agree with transactions last week?"]}
                ans.message = ans.clarification["question"]
                if problems:
                    ans.steps.append(Step("validate", "fail", "; ".join(problems)))
                return ans
            ans.steps.append(Step("validate", "ok", f"intent={plan.intent}, metric={plan.metric}"))

            if plan.intent != "free_sql":
                w = self._clip(plan.window, ans)
                if w is None:
                    ans.status = "clarify"
                    ans.assumptions = []
                    ans.message = (f"No data for {plan.window.label()}. The dataset covers "
                                   f"{self.data_start:%d %b} - {self.data_end:%d %b %Y}.")
                    ans.clarification = {"question": ans.message,
                                         "options": ["Use the last week in the data instead",
                                                     "Use October vs September"]}
                    return ans
                plan.window = w
                plan.baseline = self._clip(plan.baseline, ans)
                ans.plan = plan.to_dict()
                self._attach_dq(ans, plan)

            handler = {"diagnose_change": self._diagnose, "breakdown": self._breakdown,
                       "trend": self._trend, "compare_sources": self._sources,
                       "free_sql": self._free_sql}[plan.intent]
            handler(plan, role, ans)
            if ans.unaddressed_terms:
                terms = ", ".join(f"'{t}'" for t in ans.unaddressed_terms)
                ans.caveats.insert(0, f"The question mentions {terms}, which this answer may not apply. "
                                      "Check the plan in the Trace tab before relying on it.")
                ans.confidence = {"level": "low", "reasons": [f"Terms not accounted for in the plan: {terms}."]
                                  + ans.confidence.get("reasons", [])}
            ans.citation = self._citation(plan)
        except AccessDenied as e:
            ans.caveats, ans.facts = [], []
            ans.status = "denied"
            ans.denial_rule = e.rule
            ans.message = str(e)
            ans.steps.append(Step("guard", "denied", f"[{e.rule}] {e}"))
            ans.recommendations = ["Request access through the data owner, or ask a question within your scope."]
        except UnsafeQuery as e:
            ans.caveats, ans.facts = [], []
            ans.status = "blocked"
            ans.denial_rule = e.rule
            ans.message = f"Query blocked by safety guard: {e}"
            ans.steps.append(Step("guard", "blocked", f"[{e.rule}] {e}"))
        except QueryTimeout as e:
            ans.status = "error"
            ans.message = f"Query cancelled: {e}. Try a narrower time window or fewer dimensions."
        except Exception as e:  # noqa: BLE001
            ans.status = "error"
            ans.message = f"{type(e).__name__}: {e}"
            ans.steps.append(Step("execute", "error", ans.message))
        finally:
            ans.total_ms = (time.perf_counter() - t0) * 1000
        return ans

    def run_sql(self, sql: str, role_key: str, *, user: str = "") -> Answer:
        """SQL playground: hand-written SQL gets exactly the same guard as model SQL."""
        role = ROLES[role_key]
        ans = Answer(question=sql, role=role.label, planner_used="(hand-written SQL)", user=user)
        t0 = time.perf_counter()
        try:
            self._free_sql(Plan(intent="free_sql", sql=sql), role, ans)
            ans.caveats = ["Hand-written SQL: passed the safety guard; interpretation is yours."]
        except AccessDenied as e:
            ans.status, ans.denial_rule, ans.message = "denied", e.rule, str(e)
        except UnsafeQuery as e:
            ans.status, ans.denial_rule, ans.message = "blocked", e.rule, f"Query blocked by safety guard: {e}"
        except QueryTimeout as e:
            ans.status, ans.message = "error", f"Query cancelled: {e}"
        except Exception as e:  # noqa: BLE001
            ans.status, ans.message = "error", f"{type(e).__name__}: {str(e).splitlines()[0]}"
        ans.total_ms = (time.perf_counter() - t0) * 1000
        return ans

    # ------------------------------------------------------------ shared
    def _attach_dq(self, ans: Answer, plan: Plan) -> None:
        s = min(plan.window.start, plan.baseline.start) if plan.baseline else plan.window.start
        e = plan.window.end
        hits = [f for f in self.dq if f.status != "pass" and f.overlaps(s, e)]
        for f in hits:
            if f.check == "duplicate_records" and plan.use_raw:
                ans.caveats.append("You asked for RAW data: " + f.summary + " Raw figures are NOT de-duplicated.")
            elif f.check == "source_disagreement" and plan.intent != "compare_sources":
                continue  # only relevant if comparing sources or using settlements
            else:
                ans.caveats.append(f"Data quality ({f.check.replace('_', ' ')}): {f.summary} {f.handling}")
        ans.steps.append(Step("data_quality", "warn" if hits else "ok",
                              f"{len(hits)} finding(s) overlap the analysis window" if hits
                              else "no data-quality issues in window"))

    def _citation(self, plan: Plan) -> dict:
        return {
            "datasets": "transactions_raw (as ingested)" if plan.use_raw else
                        "transactions (de-duplicated) + merchants, banks, payment_methods"
                        + (" + settlements" if plan.intent == "compare_sources" else ""),
            "metric": f"{METRICS[plan.metric].label}: {METRICS[plan.metric].definition}" if plan.metric in METRICS else "",
            "filters": plan.filters or "none",
            "window": plan.window.label() if plan.window else "",
            "baseline": plan.baseline.label() if plan.baseline else "",
        }

    def _fact(self, ans: Answer, text: str, *recs: QueryRecord) -> None:
        ans.facts.append({"text": text, "evidence": [r.query_id for r in recs]})

    def _outcome_split(self, plan: Plan, role: Role, ans: Answer, d: float) -> None:
        """Technical vs business: the first cut in any success-rate review. Technical failures (timeouts,
        downtime) are ours or a partner's to fix; business declines (funds, drop-off, risk/issuer policy)
        are not. Splits the change exactly: dSR = -(d technical + d business + d pending)."""
        p2 = copy.deepcopy(plan)
        p2.metric = "attempts"
        df, rec = self._run(ans, role, compiler.two_period_segments(p2, ["decline_type"]),
                            "Outcome split: technical failures vs business declines")
        pv = df.pivot_table(index="segment", columns="period", values="denom", aggfunc="sum", fill_value=0)
        if not {"baseline", "current"} <= set(pv.columns):
            return
        tot = pv.sum()
        share = pv / tot
        get = lambda k, per: float(share.loc[k, per]) if k in share.index else 0.0  # noqa: E731
        d_tech = get("technical failure", "current") - get("technical failure", "baseline")
        d_bus = get("business decline", "current") - get("business decline", "baseline")
        d_pen = get("pending", "current") - get("pending", "baseline")

        def tsr(per):
            succ = float(pv.loc["success", per]) if "success" in pv.index else 0.0
            bus = float(pv.loc["business decline", per]) if "business decline" in pv.index else 0.0
            return succ / max(float(tot[per]) - bus, 1)
        ans.tables["Outcome split (share of attempts)"] = (share * 100).round(2).reset_index()
        sign = -1 if plan.metric != "failure_rate" else 1
        parts = (f"technical failures {pts(sign * d_tech)}, business declines {pts(sign * d_bus)}"
                 + (f", pending {pts(sign * d_pen)}" if plan.metric != "failure_rate" else ""))
        self._fact(ans, f"By outcome type, the change splits into {parts}. Technical failure rate "
                   f"{get('technical failure', 'baseline'):.2%} → {get('technical failure', 'current'):.2%}; "
                   f"business decline rate {get('business decline', 'baseline'):.2%} → "
                   f"{get('business decline', 'current'):.2%}; technical success rate (excludes business "
                   f"declines) {tsr('baseline'):.2%} → {tsr('current'):.2%}.", rec)
        total = abs(d_tech) + abs(d_bus)
        if total < 0.001:
            return
        if abs(d_tech) >= 2 * abs(d_bus):
            ans.hypotheses.append("The movement is mainly technical failures (timeouts, issuer or network "
                                  "downtime), not customer behaviour. That can be a genuine outage or more traffic "
                                  "on routes that time out more often (a mix effect); the segment analysis below "
                                  "separates the two.")
            ans.recommendations.insert(0, "The lever is technical: routing, retries and issuer/acquirer uptime for the "
                                          "segments identified below, whether the cause is mix or a real outage.")
        elif abs(d_bus) >= 2 * abs(d_tech):
            ans.hypotheses.append("The movement is mainly business declines (insufficient funds, customer "
                                  "drop-off, risk or issuer policy): not an infrastructure problem.")
            ans.recommendations.insert(0, "Look at customer mix, checkout drop-off and risk-rule changes; "
                                          "infrastructure changes will not recover these declines.")
        else:
            ans.hypotheses.append("Both technical failures and business declines moved; treat them separately "
                                  "(engineering for the technical part, product and risk for the business part).")
        ans.steps.append(Step("outcome_split", "ok", parts))

    def _premise_check(self, ans: Answer, label: str, delta: float, rec: QueryRecord) -> None:
        """Say so plainly when the question assumes the wrong direction ("why did GMV drop" when it rose)."""
        asked = premise_direction(ans.question)
        actual = "down" if delta < 0 else "up" if delta > 0 else None
        if asked and actual and asked != actual:
            said = "dropped" if asked == "down" else "rose"
            did = "rose" if actual == "up" else "fell"
            name = label if label[:3].isupper() else label.lower()
            note = f"Premise check: the question assumes {name} {said}, but it actually {did}."
            ans.headline = note + " " + ans.headline
            self._fact(ans, note, rec)
            ans.steps.append(Step("premise_check", "warn", note))

    # ------------------------------------------------------------ diagnose
    def _diagnose(self, plan: Plan, role: Role, ans: Answer) -> None:
        m = METRICS[plan.metric]
        w, b = plan.window, plan.baseline
        head_sql = compiler.two_period_segments(plan, [])
        head, q_head = self._run(ans, role, head_sql, "Headline: metric in both periods")
        hp = head.set_index("period")
        if not {"current", "baseline"} <= set(hp.index):
            raise RuntimeError("One of the periods has no data in your scope.")
        na, sa = float(hp.loc["baseline", "denom"]), float(hp.loc["baseline", "numer"])
        nb, sb = float(hp.loc["current", "denom"]), float(hp.loc["current", "numer"])

        trend_plan = copy.copy(plan)
        tr, q_tr = self._run(ans, role, compiler.daily_trend(trend_plan, Window(b.start, w.end)),
                             "Daily trend across both periods")
        ans.tables["Daily trend"] = tr
        ans.charts.append({"kind": "trend", "title": f"Daily {m.label.lower()}", "df": tr, "x": "day",
                           "y": m.key, "metric": m.key, "windows": {"baseline": b, "current": w}})

        if not m.is_rate:
            self._diagnose_level(plan, role, ans, na, nb, sa, sb, q_head)
            return

        ra, rb = sa / na, sb / nb
        d, lo, hi = overall_change_ci(na, sa, nb, sb)
        direction = "fell" if d < 0 else "rose"
        ans.headline = (f"{m.label} {direction} from {fmt(m.key, ra)} to {fmt(m.key, rb)} "
                        f"({pts(d)}) between {b.label()} and {w.label()}.")
        self._premise_check(ans, m.label, d, q_head)
        self._fact(ans, ans.headline + f" 95% CI for the change: {pts(lo)} to {pts(hi)}; "
                   f"attempts {na:,.0f} → {nb:,.0f} ({(nb / na - 1):+.0%}).", q_head)
        ans.steps.append(Step("headline", "ok", ans.headline))
        if plan.metric in ("success_rate", "failure_rate", "technical_success_rate"):
            self._outcome_split(plan, role, ans, d)

        if abs(d) < 0.002 or (lo < 0 < hi):
            ans.hypotheses.append("The change is within normal noise; there is no material movement to explain.")
            ans.confidence = {"level": "high", "reasons": ["Change is not statistically significant."]}
            return

        # scan dimensions
        decs: dict[str, Decomposition] = {}
        recs: dict[str, QueryRecord] = {}
        dims_to_scan = [(x,) for x in ["method", "bank", "category", "merchant", "city_tier"]] + COMBO_DIMS
        for dims in dims_to_scan:
            if any(dd in plan.filters for dd in dims):
                continue
            key = " × ".join(DIMENSIONS[x].label for x in dims)
            df, rec = self._run(ans, role, compiler.two_period_segments(plan, list(dims)),
                                f"Segment split by {key}")
            decs[key] = decompose(df, key)
            recs[key] = rec
        ans.steps.append(Step("driver_scan", "ok", f"decomposed change across {len(decs)} segmentations"))

        # primary view = segmentation where mix explains most (same sign as change)
        primary_key = max(decs, key=lambda k: decs[k].mix_total * (1 if d > 0 else -1))
        P = decs[primary_key]
        mix_pts, rate_pts = P.mix_total, P.rate_total
        mix_dominant = abs(mix_pts) >= abs(rate_pts) and P.mix_total * d > 0
        lfl = P.like_for_like_b

        summary = pd.DataFrame([{
            "segmentation": k, "mix effect (pts)": v.mix_total * 100, "rate effect (pts)": v.rate_total * 100,
            "mix share of change": v.mix_share} for k, v in decs.items()]).sort_values(
            "mix effect (pts)", ascending=d > 0)
        ans.tables["Mix vs rate by segmentation"] = summary

        self._fact(ans, f"Split by {primary_key}: traffic-mix shift accounts for {pts(mix_pts)} "
                   f"({P.mix_share:.0%} of the change) and within-segment rate changes for {pts(rate_pts)}.",
                   recs[primary_key])
        self._fact(ans, f"Like-for-like: at the baseline traffic mix, the current period's {m.label.lower()} "
                   f"would be {fmt(m.key, lfl)} ({pts(lfl - ra)} vs baseline) instead of {fmt(m.key, rb)}.",
                   recs[primary_key])

        seg = P.segments.copy()
        seg_tbl = seg[["segment", "denom_baseline", "denom_current", "w_a", "w_b", "r_a", "r_b",
                       "mix_effect", "rate_effect", "total_effect", "z_rate"]].copy()
        ans.tables[f"Segment contributions ({primary_key})"] = seg_tbl
        ans.charts.append({"kind": "contrib", "title": f"What moved the change - by {primary_key}",
                           "df": seg_tbl, "delta": d})

        # biggest mix contributors in primary view
        mix_top = seg[seg.mix_effect * d > 0].reindex(
            seg.mix_effect.abs().sort_values(ascending=False).index).dropna(subset=["mix_effect"])
        mix_top = mix_top[(mix_top.mix_effect * d > 0) & (mix_top.mix_effect.abs() >= max(0.001, 0.1 * abs(d)))].head(2)
        for r in mix_top.itertuples():
            self._fact(ans, f"{r.segment}: share of attempts {r.w_a:.1%} → {r.w_b:.1%}; its {m.label.lower()} "
                       f"{fmt(m.key, r.r_a)} → {fmt(m.key, r.r_b)} vs overall baseline {fmt(m.key, ra)}. "
                       f"Mix effect {pts(r.mix_effect)}.", recs[primary_key])

        # where did the extra volume come from (merchant view)
        surge = None
        if "Merchant" in decs:
            ms = decs["Merchant"].segments.copy()
            ms["dw"] = ms.w_b - ms.w_a
            top = ms.sort_values("dw", ascending=False).iloc[0]
            if top.dw > 0.05:
                surge = top
                self._fact(ans, f"Volume shift: {top.segment} went from {top.denom_baseline:,.0f} to "
                           f"{top.denom_current:,.0f} attempts (share {top.w_a:.1%} → {top.w_b:.1%}), "
                           f"at {fmt(m.key, top.r_b)}.", recs["Merchant"])

        # genuine rate drivers: significant, material, same sign; most specific first, dedupe parents
        thresh = max(0.0005, 0.05 * abs(d))
        # Only look for rate drivers in the primary (most mix-explanatory) segmentation: in a
        # coarser one, a mix shift *inside* a segment masquerades as that segment's rate change.
        rate_view = primary_key if not primary_key.startswith("Merchant") else "Bank × Payment method"
        cands = []
        for k, dec in decs.items():
            if k != rate_view:
                continue
            s = dec.segments
            s = s[(s.rate_effect * d > 0) & (s.rate_effect.abs() >= thresh) & (s.z_rate.abs() >= 3)]
            for r in s.itertuples():
                cands.append((abs(r.z_rate), k, r))
        cands.sort(key=lambda x: -x[0])
        picked = []
        for _, k, r in cands:
            parts = set(str(r.segment).split(" · "))
            if any(parts <= set(str(p.segment).split(" · ")) or set(str(p.segment).split(" · ")) <= parts
                   for _, p in picked):
                continue
            picked.append((k, r))
            if len(picked) == 2:
                break
        for k, r in picked:
            self._fact(ans, f"Genuine performance change in {r.segment} ({k}): {m.label.lower()} "
                       f"{fmt(m.key, r.r_a)} → {fmt(m.key, r.r_b)} (z = {r.z_rate:.1f}), "
                       f"rate effect {pts(r.rate_effect)}.", recs[k])
            reason_tbl = self._failure_reasons(plan, role, ans, r.segment, k)
            if reason_tbl is not None and len(reason_tbl):
                top = reason_tbl.iloc[0]
                self._fact(ans, f"In {r.segment}, top failure reason in the current period is "
                           f"{top.failure_reason} ({top.share_current:.0%} of failures, "
                           f"{top.share_baseline:.0%} in baseline).", ans.queries[-1])

        # hypotheses (clearly labelled as such)
        if mix_dominant and len(mix_top):
            seg0 = mix_top.iloc[0].segment
            msg = (f"Most of the decline is a change in which payments we processed rather than payments "
                   f"getting worse: more traffic flowed through {seg0}, a segment that was already "
                   f"below average.")
            if surge is not None:
                msg += (f" The extra traffic coincides with {surge.segment}'s volume surge - plausibly a "
                        f"sale or campaign, which this data cannot confirm.")
            ans.hypotheses.append(msg)
        elif not mix_dominant:
            ans.hypotheses.append("Most of the change is within-segment performance, i.e. payments in "
                                  "existing segments started failing more often.")
        for k, r in picked:
            ans.hypotheses.append(f"{r.segment} degradation looks like an issuer/partner-side incident "
                                  "(sudden, significant, concentrated in one segment). Needs confirmation "
                                  "from the bank or gateway status logs.")

        # recommendations
        if mix_dominant:
            ans.recommendations.append(f"Report the like-for-like rate ({fmt(m.key, lfl)}) alongside the headline "
                                       "so a mix shift is not read as a platform regression.")
            if len(mix_top):
                ans.recommendations.append(f"Look at routing for {mix_top.iloc[0].segment} (e.g. retries, alternate "
                                           "acquirer/PSP handle, smart routing) since it is structurally weak and "
                                           "now carries more volume.")
            if surge is not None:
                ans.recommendations.append(f"Confirm with the {surge.segment} account team whether a campaign is "
                                           "running and expected to continue; forecast impact if it does.")
        for k, r in picked:
            ans.recommendations.append(f"Open an incident with the partner for {r.segment}; add an alert on "
                                       "segment-level success-rate drops (z ≥ 3) to catch this sooner.")
        if any("duplicate" in c for c in ans.caveats) or any(f.check == "duplicate_records" and f.status != "pass"
                                                             and f.overlaps(b.start, w.end) for f in self.dq):
            ans.recommendations.append("Fix de-duplication in the ingestion retry path; raw dashboards built on "
                                       "the un-deduplicated table overstate failures.")

        # confidence
        reasons = [f"Overall change is significant (CI excludes 0, n = {na + nb:,.0f})."]
        level = "high"
        gaps = [f for f in self.dq if f.check == "missing_data" and f.status != "pass"]
        for nm, win in (("baseline", b), ("current", w)):
            days = sorted({pd.Timestamp(x).date() for f in gaps for x in f.affected_days
                           if win.start <= pd.Timestamp(x).date() <= win.end})
            if days:
                level = "medium"
                reasons.append(f"Incomplete data on {', '.join(f'{x:%d %b}' for x in days)} falls in the "
                               f"{nm} period, so that period's {m.label.lower()} carries a time-of-day bias.")
        if any("missing bank" in c for c in ans.caveats):
            reasons.append("Some bank_id values missing in window (small).")
        reasons.append("Driver attribution is descriptive (decomposition), not causal; causes are hypotheses.")
        ans.confidence = {"level": level, "reasons": reasons}

    def _diagnose_level(self, plan, role, ans, na, nb, sa, sb, q_head):
        """Change in a level metric (attempts, GMV): contribution of each segment to the delta."""
        m = METRICS[plan.metric]
        va, vb = sa, sb
        if m.key == "attempts":
            va, vb = na, nb
        direction = "fell" if vb < va else "rose"
        ans.headline = f"{m.label} {direction} from {fmt(m.key, va)} to {fmt(m.key, vb)} ({(vb / va - 1):+.1%})."
        self._premise_check(ans, m.label, vb - va, q_head)
        self._fact(ans, ans.headline, q_head)
        for dim in ["merchant", "method", "bank"]:
            df, rec = self._run(ans, role, compiler.two_period_segments(plan, [dim]), f"Contribution by {dim}")
            p = df.pivot_table(index="segment", columns="period",
                               values="denom" if m.key == "attempts" else "numer", aggfunc="sum", fill_value=0)
            p["delta"] = p.get("current", 0) - p.get("baseline", 0)
            p = p.sort_values("delta", key=abs, ascending=False).reset_index()
            ans.tables[f"Contribution by {DIMENSIONS[dim].label}"] = p
            top = p.iloc[0]
            self._fact(ans, f"Largest contributor by {DIMENSIONS[dim].label.lower()}: {top.segment} "
                       f"({fmt(m.key, top.delta)} of the {fmt(m.key, vb - va)} change).", rec)
        ans.confidence = {"level": "high", "reasons": ["Direct sums; no modelling involved."]}

    def _failure_reasons(self, plan: Plan, role: Role, ans: Answer, segment: str, seg_key: str):
        dims = [d.strip() for d in seg_key.split("×")]
        label_to_key = {DIMENSIONS[k].label: k for k in DIMENSIONS}
        keys = [label_to_key.get(x) for x in dims]
        vals = str(segment).split(" · ")
        p2 = copy.deepcopy(plan)
        for k, v in zip(keys, vals):
            if k in ("bank", "method", "merchant", "category", "city_tier"):
                if v.startswith("Unknown") or v.startswith("N/A"):
                    return None
                p2.filters = {**p2.filters, k: [v]}
        p2.metric = "failure_rate"
        sql = compiler.two_period_segments(p2, ["failure_reason"])
        df, _ = self._run(ans, role, sql, f"Failure reasons within {segment}")
        df = df[df.segment != "(not failed)"]
        pv = df.pivot_table(index="segment", columns="period", values="numer", aggfunc="sum", fill_value=0)
        for c in ("baseline", "current"):
            if c not in pv:
                pv[c] = 0
        pv["share_baseline"] = pv.baseline / max(pv.baseline.sum(), 1)
        pv["share_current"] = pv.current / max(pv.current.sum(), 1)
        pv = pv.sort_values("current", ascending=False).reset_index().rename(columns={"segment": "failure_reason"})
        ans.tables[f"Failure reasons - {segment}"] = pv
        return pv

    # ------------------------------------------------------------ breakdown
    def _breakdown(self, plan: Plan, role: Role, ans: Answer) -> None:
        m = METRICS[plan.metric]
        dims = plan.dimensions
        cur, q1 = self._run(ans, role, compiler.metric_by(plan, dims, plan.window),
                            f"{m.label} by {', '.join(dims) or 'total'} ({plan.window.label()})")
        if not len(cur):
            ans.headline = "No rows match these filters in your scope."
            self._fact(ans, ans.headline, q1)
            return
        recs = [q1]
        if plan.baseline is not None:
            base, q2 = self._run(ans, role, compiler.metric_by(plan, dims, plan.baseline),
                                 f"{m.label} by {', '.join(dims) or 'total'} ({plan.baseline.label()})")
            recs.append(q2)
            if dims:
                cur = cur.merge(base.rename(columns={m.key: f"{m.key}_baseline", "attempts": "attempts_baseline"}),
                                on=dims, how="left")
            else:
                cur[f"{m.key}_baseline"] = base[m.key].values[:1]
                cur["attempts_baseline"] = base["attempts"].values[:1]
            cur["change"] = cur[m.key] - cur[f"{m.key}_baseline"]
        ans.tables[f"{m.label} by {', '.join(dims) or 'total'}"] = cur

        if not dims:
            v = cur[m.key].iloc[0]
            ans.headline = f"{m.label} for {plan.window.label()}: {fmt(m.key, v)} over {cur.attempts.iloc[0]:,.0f} attempts."
            if "change" in cur:
                ans.headline += f" Baseline ({plan.baseline.label()}): {fmt(m.key, cur[f'{m.key}_baseline'].iloc[0])}."
            self._fact(ans, ans.headline, *recs)
            ans.confidence = {"level": "medium" if plan.use_raw else "high", "reasons": [
                "Raw table includes duplicate rows." if plan.use_raw else "Direct aggregate from the de-duplicated table."]}
            return

        ans.charts.append({"kind": "bar", "title": f"{m.label} by {' × '.join(DIMENSIONS[d].label for d in dims)}",
                           "df": cur, "x": dims[0], "y": m.key, "metric": m.key, "color": dims[1] if len(dims) > 1 else None})
        s = cur.sort_values(m.key, ascending=False)
        lab = lambda r: " · ".join(str(r[d]) for d in dims)  # noqa: E731
        best, worst = s.iloc[0], s.iloc[-1]
        ans.headline = (f"{m.label} by {', '.join(DIMENSIONS[d].label.lower() for d in dims)} "
                        f"for {plan.window.label()}: highest {lab(best)} ({fmt(m.key, best[m.key])}), "
                        f"lowest {lab(worst)} ({fmt(m.key, worst[m.key])}).")
        self._fact(ans, ans.headline, q1)
        if "change" in cur and m.is_rate:
            mover = cur.loc[cur.change.abs().idxmax()]
            self._fact(ans, f"Biggest move vs {plan.baseline.label()}: {lab(mover)} "
                       f"{fmt(m.key, mover[f'{m.key}_baseline'])} → {fmt(m.key, mover[m.key])} ({pts(mover.change)}).",
                       *recs)
        small = cur[cur.attempts < 200]
        if m.is_rate and len(small):
            ans.caveats.append(f"{len(small)} segment(s) have fewer than 200 attempts; their rates are noisy.")
        ans.confidence = {"level": "high", "reasons": ["Direct aggregates; no modelling."]}

    # ------------------------------------------------------------ trend
    def _trend(self, plan: Plan, role: Role, ans: Answer) -> None:
        m = METRICS[plan.metric]
        df, rec = self._run(ans, role, compiler.daily_trend(plan, plan.window), "Daily trend")
        ans.tables["Daily trend"] = df
        ans.charts.append({"kind": "trend", "title": f"Daily {m.label.lower()}", "df": df, "x": "day",
                           "y": m.key, "metric": m.key, "windows": {}})
        hi, lo = df.loc[df[m.key].idxmax()], df.loc[df[m.key].idxmin()]
        ans.headline = (f"Daily {m.label.lower()} {plan.window.label()}: ranged from {fmt(m.key, lo[m.key])} "
                        f"({pd.Timestamp(lo.day):%d %b}) to {fmt(m.key, hi[m.key])} ({pd.Timestamp(hi.day):%d %b}).")
        self._fact(ans, ans.headline, rec)
        if len(df) >= 14 and m.is_rate:
            first, last = df.head(7)[m.key].mean(), df.tail(7)[m.key].mean()
            self._fact(ans, f"Last 7 days average {fmt(m.key, last)} vs first 7 days {fmt(m.key, first)} "
                       f"({pts(last - first)}).", rec)
            if last - first < -0.01:
                ans.recommendations.append("Ask 'why did success rate drop last week?' to decompose the decline.")
        ans.confidence = {"level": "high", "reasons": ["Direct daily aggregates."]}

    # ------------------------------------------------------------ sources
    def _sources(self, plan: Plan, role: Role, ans: Answer) -> None:
        df, rec = self._run(ans, role, compiler.source_reconciliation(plan),
                            "Reconcile transactions vs settlements")
        df["diff_success"] = df.st_success - df.tx_success
        ans.tables["Transactions vs settlements"] = df
        ans.charts.append({"kind": "sources", "title": "Success rate: transactions vs settlements", "df": df})
        tx_sr = df.tx_success.sum() / df.tx_attempts.sum()
        st_sr = df.st_success.sum() / df.st_attempts.sum()
        ans.headline = (f"The two sources disagree: success rate is {tx_sr:.2%} in transactions but {st_sr:.2%} "
                        f"in settlements for {plan.window.label()} ({pts(st_sr - tx_sr)}).")
        self._fact(ans, ans.headline, rec)
        expl = df.diff_success.sum() / max(df.tx_pending.sum(), 1)
        self._fact(ans, f"Settlements count {df.diff_success.sum():,.0f} more successes; transactions show "
                   f"{df.tx_pending.sum():,.0f} payments as PENDING. The gap equals {expl:.0%} of pending payments.", rec)
        att_gap = (df.tx_attempts - df.st_attempts).abs().sum()
        self._fact(ans, f"Attempt counts match (total absolute difference {att_gap:,.0f}) once transactions "
                   "are de-duplicated.", rec)
        ans.hypotheses.append("The gap is explained by PENDING payments that settled later: transactions record "
                              "the real-time outcome, settlements record the final one.")
        ans.recommendations += [
            "Use transactions for customer payment-experience metrics (real-time success).",
            "Use settlements for money and reconciliation questions (final outcome).",
            "Label dashboards with which source and denominator they use.",
        ]
        ans.confidence = {"level": "high", "reasons": ["Both sources queried directly over identical days."]}

    # ------------------------------------------------------------ free sql
    def _free_sql(self, plan: Plan, role: Role, ans: Answer) -> None:
        df, rec = self._run(ans, role, plan.sql, "Model-written SQL (validated by guard)")
        ans.tables["Result"] = df
        if len(df) == 1 and 0 < len(df.columns) <= 4:
            def show(col, v):
                if isinstance(v, bool) or not isinstance(v, numbers.Real) or pd.isna(v):
                    return str(v)
                v = float(v)  # numpy / DuckDB numeric types included
                if 0 <= v <= 1 and v % 1 and re.search(r"share|rate|pct|ratio|fraction|percent", col):
                    return f"{v:.2%}"
                return f"{v:,.2f}" if v % 1 else f"{v:,.0f}"
            ans.headline = "; ".join(f"{c.replace('_', ' ')} = {show(c, df.iloc[0][c])}" for c in df.columns) + "."
        else:
            ans.headline = f"Query returned {len(df)} row(s)."
        self._fact(ans, ans.headline + " Values are shown exactly as returned.", rec)
        try:
            where = sqlglot.parse_one(plan.sql, read="duckdb").find(exp.Where)
        except sqlglot.errors.ParseError:
            where = None
        if where is not None:
            conds, stack = [], [where.this]
            while stack:
                node = stack.pop()
                if isinstance(node, exp.And):
                    stack += [node.right, node.left]
                else:
                    conds.append(node.sql(dialect="duckdb"))
            self._fact(ans, "Conditions applied by this query: " + "; ".join(f"`{c}`" for c in conds) + ".", rec)
        vague = [g.split(": ", 1)[-1].strip("'") for g in unmodelled_conditions(ans.question)
                 if g.startswith("vague threshold")]
        if vague:
            stem = re.split(r"[- ]", vague[0])[0]
            defs = ([a for a in plan.assumptions if re.search(r"\d", a)
                     and re.search(rf"{re.escape(stem)}|amount|hour|value|ticket", a, re.I)]
                    or [a for a in plan.assumptions if re.search(r"\d", a)])
            if defs:
                ans.headline += " Definition used: " + "; ".join(defs) + "."
                ans.facts[-1]["text"] = ans.headline + " Values are shown exactly as returned."
                ans.caveats.append(f"'{vague[0]}' has no standard definition in this dataset. The model chose: "
                                   + "; ".join(defs) + ". A different cut-off would give a different answer.")
            else:
                ans.caveats.append(f"'{vague[0]}' has no standard definition and the model did not state the one "
                                   "it used. Read the SQL in the Evidence tab before relying on this number.")
        ans.caveats.append("This SQL was written by the language model (no catalog plan fitted). It passed the "
                           "safety guard, but check that it answers your question.")
        ans.confidence = {"level": "medium", "reasons": ["Model-written query; logic not template-verified."]}
