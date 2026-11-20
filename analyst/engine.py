"""
Orchestrator: question -> plan -> validate -> guard -> execute -> analyse -> answer.

Every number in an answer comes from an executed, logged query; every fact carries the
query ids that support it. The LLM (if configured) only produces the plan.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from . import compiler
from .dq import DQFinding, run_checks
from .drivers import Decomposition, decompose, overall_change_ci
from .executor import Executor, QueryRecord, QueryTimeout
from .guard import SqlGuard, UnsafeQuery
from .llm import LLMCall, LLMPlanner
from .plan import Plan, Window, validate_plan
from .rbac import ROLES, AccessDenied, Role
from .rule_planner import RulePlanner
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


class AnalystEngine:
    def __init__(self, db_path: str, llm_provider: str | None = None):
        self.ex = Executor(db_path)
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
                              role=role.key, checks=g.checks)
        ans.queries.append(rec)
        return df, rec

    # ------------------------------------------------------------ planning
    def make_plan(self, question: str, ans: Answer, mode: str) -> Plan:
        t0 = time.perf_counter()
        if mode in ("auto", "llm") and self.llm.enabled:
            res = self.llm.plan(question)
            ans.llm_calls.extend(res.calls)
            if res.plan is not None:
                ans.planner_used = f"LLM ({self.llm.provider}:{self.llm.model})"
                ans.steps.append(Step("plan", "ok", f"{ans.planner_used}, {len(res.calls)} call(s)",
                                      (time.perf_counter() - t0) * 1000))
                return res.plan
            reason = "; ".join(res.errors) or "no plan"
            if mode == "llm":
                raise RuntimeError(f"LLM planner failed: {reason}")
            ans.steps.append(Step("plan", "fallback", f"LLM planner failed ({reason}); "
                                  "fell back to deterministic planner"))
            ans.caveats.append("The language model was unavailable or returned an invalid plan, so the "
                               "deterministic planner was used. Supported question types still work.")
        plan = self.rules.plan(question)
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
    def ask(self, question: str, role_key: str = "head_of_payments", planner_mode: str = "auto") -> Answer:
        t0 = time.perf_counter()
        role = ROLES[role_key]
        ans = Answer(question=question, role=role.label)
        try:
            plan = self.make_plan(question, ans, planner_mode)
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

    def run_sql(self, sql: str, role_key: str) -> Answer:
        """SQL playground: hand-written SQL gets exactly the same guard as model SQL."""
        role = ROLES[role_key]
        ans = Answer(question=sql, role=role.label, planner_used="(hand-written SQL)")
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
        self._fact(ans, ans.headline + f" 95% CI for the change: {pts(lo)} to {pts(hi)}; "
                   f"attempts {na:,.0f} → {nb:,.0f} ({(nb / na - 1):+.0%}).", q_head)
        ans.steps.append(Step("headline", "ok", ans.headline))

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
        if any(c.startswith("Data quality (missing data)") for c in ans.caveats):
            level = "medium"
            reasons.append("An incomplete day falls inside the analysis window.")
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
        ans.headline = f"{m.label} changed from {fmt(m.key, va)} to {fmt(m.key, vb)} ({(vb / va - 1):+.1%})."
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
        ans.headline = f"Query returned {len(df)} row(s)."
        self._fact(ans, ans.headline + " Values are shown exactly as returned.", rec)
        ans.caveats.append("This SQL was written by the language model (no catalog plan fitted). It passed the "
                           "safety guard, but check that it answers your question.")
        ans.confidence = {"level": "medium", "reasons": ["Model-written query; logic not template-verified."]}
