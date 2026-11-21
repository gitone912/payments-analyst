# Payments AI Business Analyst

Ask a question about payment performance in plain English. The system plans an analysis,
validates the SQL, runs it against payments data, breaks the change down by segment,
and answers with **facts (with query evidence), hypotheses, recommendations, and caveats**,
each kept separate.

> "Why did success rate drop last week?"
> → Success rate fell 87.51% → 84.92% (−2.59 pts, CI −3.09 to −2.10). **86% of it is a
> traffic-mix shift**: Delta Co-op Bank · UPI went from 2.9% to 14.4% of volume (driven by
> QuickTix: 390 → 10,871 attempts) while its own success rate stayed flat at ~69%.
> At last week's mix the rate would be 87.18%. Separately, **Coastal Bank · NetBanking
> genuinely degraded** (87.8% → 81.4%, z = −3.5; top failure reason BANK_DOWNTIME).
> Caveats: 411 duplicate rows on 20–22 Oct (de-duplicated), incomplete data on 14 Oct.

## Quick start

```bash
pip install -r requirements.txt
python data/generate_data.py            # builds data/payments.duckdb (~285k txns, seeded)
python run_app.py                       # UI at http://localhost:8501 (sign in: see below)
python evals/run_evals.py               # 40 known-answer / safety checks
python -m pytest -q tests               # 23 unit tests
python cli.py "why did success rate drop last week?" --sql
python cli.py "why did success rate drop last week?" --then "now break that down by bank"
```

`run_app.py` wraps `streamlit run app.py`. On Windows it avoids the harmless
`ConnectionResetError [WinError 10054]` tracebacks when a browser tab closes, and Ctrl+C always
returns within a few seconds instead of hanging on "Stopping...". If `streamlit` is "not recognized"
on Windows, use `python -m streamlit run app.py` (or `python run_app.py`) instead.

Optional LLM planner (the system runs fully without one). Easiest: copy `.env.example` to `.env`,
put your key in it and restart; the app, CLI and LLM evals read it automatically (`.env` is git-ignored;
variables already set in the terminal take precedence). Check it with `python -m analyst.llm_check`: one real call per
configured model, with a plain reason if it fails (no key, key rejected, model retired, rate limit). Or set the variables yourself:

```bash
export LLM_PROVIDER=gemini GEMINI_API_KEY=...            # default model: gemini-3.5-flash-lite
export LLM_PROVIDER=openai OPENAI_API_KEY=... [OPENAI_BASE_URL=... for Groq/OpenRouter/Ollama]
export LLM_MODEL=...                                     # optional override
export LLM_MIN_INTERVAL_S=4.5                            # optional pacing for free-tier keys (15 req/min)
python evals/run_evals.py --planner llm --paraphrases    # evals against the live model
```

The Gemini key is sent in the `x-goog-api-key` header, never in the URL, so it can't leak into logs or tracebacks.
Older defaults (`gemini-2.0-flash`, `gemini-2.5-flash`) now return 404 for new keys.

### Sign-in

The role comes from the signed-in account, never from a control the user can change.

| Demo account | Password | Role |
|---|---|---|
| `head.payments` | `demo-head` | Head of Payments (all merchants, no PII) |
| `risk.ops` | `demo-risk` | Risk Ops (PII visible) |
| `kam.a` | `demo-kam` | KAM Portfolio A (own merchants only) |

These are **demo accounts** in `config/users.demo.json` (salted PBKDF2-SHA256 hashes). For anything
real, point `USERS_FILE` at your own file (`python -m analyst.auth add <user> <role>` prompts for a
password), or run behind an SSO proxy with `AUTH_MODE=header`: the proxy authenticates the user and
sets `X-Forwarded-Email` (configurable via `AUTH_HEADER`); the role is still looked up from the users
file. Header mode is only safe when the app is reachable exclusively through that proxy.

Every query is appended to `logs/audit.jsonl` (user, role, time, SQL, outcome; path via `AUDIT_LOG`).
Users see only their own entries in the Session log tab.

Deep link for demos: `http://localhost:8501/?q=Why did success rate drop last week?`

## Interface

A product-style UI built only on Streamlit's native theming (no extra UI dependencies):

- **Navigation:** sidebar pages *Ask*, *SQL console* and *Audit log*; assistant settings
  (planner, LLM status, outage simulation), data health and the signed-in user below them.
- **Answers read like a short analyst report:** headline with a confidence badge and a status
  line (planner, queries, time, model cost), then Findings (each with the query it came from),
  Likely causes, Recommended actions, Caveats and Method; Charts, Queries, Data and Trace tabs.
- **Design system:** `.streamlit/config.toml` (Inter, one indigo accent, neutral greys, light and
  dark palettes, compact type scale) and `ui/theme.py` (one stylesheet for the app's own elements;
  no selectors on Streamlit internals). Components in `ui/components.py`, screens in `ui/pages.py`,
  the answer view in `ui/answer.py`; `app.py` only wires them together.
- Why no shadcn/antd component libraries: `streamlit-antd-components` has had no release since
  January 2024; `streamlit-shadcn-ui` renders each component in its own iframe, which does not follow
  the app's light/dark theme or font. Streamlit's built-in Material icons replace Lucide.

## Architecture

```
question
  │
  ▼
Planner ── LLM (JSON plan, validated against catalog, 1 retry with error feedback)
  │        └─ on outage / repeated malformed output → deterministic rule planner
  ▼
Plan validation (known metrics/dimensions/filters, time window inside data, ambiguity → clarify)
  ▼
Compiler (plan → SQL over *logical* tables using the semantic layer)
  ▼
SQL Guard (sqlglot AST): read-only · single statement · allow-listed functions only ·
           catalog tables only · RBAC (tables, PII columns, merchant scope) ·
           no cartesian joins / bounded fact scans · row-level scope rewrite · auto LIMIT
  ▼
Executor (read-only DuckDB, one cursor per query, 8s timeout, result cache, persistent audit log)
  ▼
Analysis: headline + CI → mix/rate decomposition across 7 segmentations →
          significance-filtered rate drivers → failure-reason drill-down
  ▼
Answer: facts[evidence query ids] · hypotheses · recommendations · DQ caveats ·
        assumptions · confidence · citation (dataset, metric definition, window, filters)
```

| Module | Role |
|---|---|
| `data/generate_data.py` | Synthetic data with planted scenarios (documented in the file header) |
| `analyst/semantic.py` | Metric & dimension catalog; one definition of "success rate" |
| `analyst/rbac.py` | Roles: Head of Payments, Risk Ops (PII), KAM Portfolio A (row-scoped) |
| `analyst/guard.py` | SQL safety + access control + scope rewrite |
| `analyst/llm.py` | Provider abstraction, retries, token/latency/cost tracking |
| `analyst/rule_planner.py` | Deterministic planner and fallback |
| `analyst/drivers.py` | Exact mix/rate decomposition + z-scores |
| `analyst/dq.py` | Duplicates, gaps, missing dimension, source reconciliation, pending share |
| `analyst/engine.py` | Orchestration and answer composition |

### Key design decisions

- **The LLM plans; code computes.** The model only produces a JSON plan referencing the
  catalog (or, as a last resort, a single SELECT). Numbers come from executed SQL and the
  narrative is templated from those results, so the model cannot invent a figure.
- **Mix vs rate decomposition.** `ΔR = Σ Δwᵢ(r̄ᵢ − Rᴬ) + Σ Δrᵢ w̄ᵢ`, exact (unit-tested).
  The primary view is the segmentation where mix explains the most. Rate drivers are only
  taken from that view, because at a coarser level a mix shift *inside* a segment looks
  like that segment's rate dropping (e.g. "UPI got worse"). The eval suite checks this.
- **Technical vs business first.** Failures are classified once in the semantic layer: technical
  (TIMEOUT, BANK_DOWNTIME) vs business declines (insufficient funds, customer drop-off, risk and
  issuer policy). Every success-rate diagnosis splits the change exactly into technical, business
  and pending parts, and reports the technical success rate (business declines left out of the
  denominator). The hypothesis and first recommendation follow from which side moved.
  `technical_success_rate`, `technical_failure_rate`, `business_decline_rate` and the
  `decline_type` dimension are also available directly.
- **Fail closed.** The SQL guard allows only listed functions (aggregate, date, math, string), so
  a dangerous DuckDB function nobody thought of is rejected; the old blocked-name list let e.g.
  `current_setting()` through. Without the LLM, a word the planner has no rule for ("premium
  customers") gets a clarifying question, not an answer to a different question. With the LLM,
  each such word must appear in the plan or its assumptions, or the answer is marked low
  confidence with a caveat. Model-written SQL must visibly apply every detected condition
  (threshold value, day of week, time of day), and the answer lists the exact filters applied.
- **Safe for concurrent users.** One engine serves every session. All per-request state (follow-up
  context, outage simulation) travels with the request, each query runs on its own DuckDB cursor
  (a timeout cancels only that query), and query IDs are numbered per answer. A threaded test
  checks that follow-up context never crosses between users.
- **Clarify rather than guess** when the comparison period or the denominator is missing.
  Lower-risk defaults (e.g. "last 7 days" for a snapshot) are applied and listed as assumptions.
  The metric-ambiguity check is enforced in code on LLM plans too. If the model quietly picks a
  measure for "how are we performing?", the engine overrides it with a clarification and logs an
  `ambiguity_guard` step.
- **Never silently drop a condition.** If the question contains something the catalog can't
  express (an amount threshold, a distinct count, a percentile), a catalog plan would answer a
  different question with full confidence. A coverage guard detects this (regex detector + the
  model's own `unsupported` list) and re-plans as model-written SQL, which goes through the same
  guard, or asks the user. It never answers while ignoring the condition.
  Word conditions count too: weekends, time of day, "high-value"/"big transactions", and new or
  returning customers. When the model picks a cut-off for a vague term, the definition goes into
  the headline and a caveat.
- **Period check on LLM plans.** The dates a model's plan (or SQL) uses must lie inside the period
  the question names. A plan that answers "weekends last week" with 18-19 Oct is sent back once
  with the error; if it is still wrong, the deterministic planner takes over.
- **Follow-ups.** "Now break that down by bank" / "what about UPI?" keep the previous question's
  metric, period, filters and analysis type (both planners). Standalone questions ignore it.
- **Premise check.** "Why did GMV drop?" when GMV rose is answered with
  "Premise check: ... it actually rose" before the numbers.
- **Provider resilience.** When the primary model is overloaded or rate-limited, the planner tries
  backup models in order (`LLM_FALLBACK_MODELS`, default `gemini-3.1-flash-lite` for Gemini). A
  short wait the provider asks for (Retry-After / `retryDelay`, at most `LLM_MAX_RETRY_WAIT_S`=5s) is
  honoured once per model; longer ones are not waited for. After 3 consecutive failures a model is
  skipped for 60s (circuit breaker, shared across users), and each request has a 20s budget
  (`LLM_REQUEST_BUDGET_S`). At most two calls per model, then the deterministic planner takes over.
  The trace shows which model answered.
- **LLM robustness**: tolerant JSON parsing (trailing commas), one retry with the validation
  errors fed back, no retry on 429/5xx (falls back immediately), and an in-process plan cache so
  re-asking a question costs no quota.
- **Access control in two layers**: an explicit deny with a reason (PII column, merchant
  literal outside portfolio), and a row-level rewrite as defence in depth.

## Evaluation results

Ground truth comes from hand-written SQL on the raw tables, not from the engine's compiler.

<!-- eval-results:start -->
<!-- generated by evals/render_results.py from evals/results/*.json; do not edit -->
| Planner | Core checks passed | Wrong answers | Provider errors | Held-out paraphrases | Run (UTC) · commit |
|---|---|---|---|---|---|
| Deterministic rules | 41/41 | 0 | 0 | 8/10 | 2026-11-21 14:59 · `a92e159-dirty` |
| LLM (live) `gemini-3.5-flash-lite` | 41/41 | 0 | 0 | 10/10 | 2026-11-21 14:37 · `a92e159-dirty` |

- Deterministic rules paraphrases not passed: "How did Card do compared with NetBanking in September?" (FAIL); "Top failure reasons last week" (FAIL)
- LLM (live): models tried in order `gemini-3.5-flash-lite` → `gemini-3.1-flash-lite`; answers came from `gemini-3.5-flash-lite` ×27, `gemini-3.1-flash-lite` ×6.
<!-- eval-results:end -->

The core suite includes questions the catalog can't express: amount thresholds, "on weekends",
and "high-value". With the LLM, the model writes SQL that matches the hand-written ground truth
exactly. Without it, the engine asks instead of answering a different question. Two follow-up
cases ("now break that down by bank", "what about UPI?") check that context carries over. The
eval output labels provider errors (429/503) as `PROVIDER`, not `FAIL`.
The eval runner does not retry: provider resilience is the app's job (below), so the numbers show
what a user would get. Exit codes: `0` all passed, `1` at least one wrong answer, `3` only provider
errors remain. The table above is generated: `python evals/run_evals.py [--planner llm] --paraphrases
--save` writes `evals/results/<planner>.json` (every case, the commit, which model answered), and
`python evals/render_results.py` rebuilds the table (`--check` fails if the README is stale).

The paraphrases were written after the prompt was finalised and are not used as prompt examples.
They show whether the planner generalises beyond the questions it was tuned against.
Runs vary slightly even at temperature 0: in earlier runs the model occasionally chose a
different plan (e.g. the "raw" question as a trend), and the provider sometimes returns 429/503. In `auto` mode those cases fall back
to the rule planner, and the fallback is shown in the trace.

## Failure scenarios covered

| Scenario | What happens | Demo |
|---|---|---|
| Missing / duplicated records | DQ checks flag 411 duplicates (20–22 Oct) and 14 Oct outage; analysis uses de-duplicated view; caveats attached when window overlaps | main question; "…raw including duplicates" |
| Request outside access scope | PII → denied for non-Risk roles; KAM asking about QuickTix → denied; KAM aggregates auto-scoped | "Show customer VPAs…", switch role to KAM |
| Two sources disagree | Transactions vs settlements reconciled; gap = 65% of PENDING resolving later; guidance on which to use | "Do settlements agree…" |
| Decline explained by mix shift | Decomposition + like-for-like rate | main question |
| Ambiguous period / denominator | Clarifying question with clickable options | "Why did success rate drop?", "How are we performing?" |
| Expensive / unsafe query | Guard blocks DELETE, stacked statements, file reads, raw-schema bypass, cartesian joins, SELECT * with PII; 8s timeout | SQL playground presets |
| Model provider unavailable / malformed output | One retry with validation errors; then fallback to rule planner, flagged in caveats & trace | sidebar "Simulate model-provider outage" |

## Assumptions, mocks and limitations

- Data is **synthetic** and seeded; bank and merchant names are fictional. "Today" = last date in the data (25 Oct 2026).
- Success rate = SUCCESS / all attempts (PENDING in denominator). The excl.-pending variant is available.
- Driver attribution is descriptive, not causal. Causes are always shown as hypotheses.
- The rule planner covers common question shapes (diagnose, breakdown, trend, source comparison).
  Anything else needs the LLM, or gets a clarification.
- Authentication is a local users file (demo accounts above) or a trusted SSO header; there is no
  built-in OIDC flow. Cost controls are heuristic (AST rules + timeout), not optimizer-based.
- The failure taxonomy (technical vs business) is a fixed mapping of this dataset's failure reasons;
  real issuer/NPCI response codes would need a maintained mapping owned by the payments team.
- The vocabulary check is deliberately strict: an unusual but harmless word makes the rule planner
  ask. Extend `_KNOWN_WORDS` in `rule_planner.py` as new phrasings come up.
- LLM cost uses list prices in `llm.py`; adjust to your contract.
