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
streamlit run app.py                    # UI at http://localhost:8501
python evals/run_evals.py               # 32 known-answer / safety checks
python -m pytest -q tests               # 9 unit tests
python cli.py "why did success rate drop last week?" --sql
```

Optional LLM planner (the system runs fully without one):

```bash
export LLM_PROVIDER=gemini GEMINI_API_KEY=...            # default model: gemini-3.5-flash-lite
export LLM_PROVIDER=openai OPENAI_API_KEY=... [OPENAI_BASE_URL=... for Groq/OpenRouter/Ollama]
export LLM_MODEL=...                                     # optional override
export LLM_MIN_INTERVAL_S=4.5                            # optional pacing for free-tier keys (15 req/min)
python evals/run_evals.py --planner llm --paraphrases    # evals against the live model
```

The Gemini key is sent in the `x-goog-api-key` header, never in the URL, so it can't leak into logs or tracebacks.
Older defaults (`gemini-2.0-flash`, `gemini-2.5-flash`) now return 404 for new keys.

Deep link for demos: `http://localhost:8501/?q=Why did success rate drop last week?`

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
SQL Guard (sqlglot AST): read-only · single statement · no file/system functions ·
           catalog tables only · RBAC (tables, PII columns, merchant scope) ·
           no cartesian joins / bounded fact scans · row-level scope rewrite · auto LIMIT
  ▼
Executor (read-only DuckDB, 8s timeout via interrupt, result cache, audit log)
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
- **Premise check.** "Why did GMV drop?" when GMV rose is answered with
  "Premise check: ... it actually rose" before the numbers.
- **LLM robustness**: tolerant JSON parsing (trailing commas), one retry with the validation
  errors fed back, no retry on 429/5xx (falls back immediately), and an in-process plan cache so
  re-asking a question costs no quota.
- **Access control in two layers**: an explicit deny with a reason (PII column, merchant
  literal outside portfolio), and a row-level rewrite as defence in depth.

## Evaluation results

Ground truth comes from hand-written SQL on the raw tables, not from the engine's compiler.

| Planner | Core suite (27 known-answer + safety checks) | Held-out paraphrases (10, plan shape) |
|---|---|---|
| Deterministic rules | 27/27 | 8/10 (misses "Card compared with NetBanking", "Top failure reasons") |
| Gemini `gemini-3.5-flash-lite` (live) | 27/27 | 8/10. Both misses were HTTP 503 from the provider; every plan the model returned was correct |

The paraphrases were written after the prompt was finalised and are not used as prompt examples.
They show whether the planner generalises beyond the questions it was tuned against.
Runs vary slightly even at temperature 0: in one of four live runs the model planned the "raw" question
as a trend, and the provider sometimes returns 429/503. In `auto` mode those cases fall back
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
- Roles are selected in the UI (no real auth). Cost controls are heuristic (AST rules + timeout), not optimizer-based.
- LLM cost uses list prices in `llm.py`; adjust to your contract.
