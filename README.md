# Payments AI Business Analyst

Ask a question about payment performance in plain English and get an answer an analyst would
sign off on: the number, **why** it moved, what to do next, and the evidence behind every claim.
Every figure comes from an executed, guarded SQL query; the language model only decides *what*
to compute, never the numbers.

> **"Why did success rate drop last week?"**
>
> Success rate fell **87.51% → 84.92% (−2.59 pts)**, 95% CI −3.09 to −2.10 pts.
> - **Mostly technical, not customers:** technical failures −2.32 pts, business declines −0.18,
>   pending −0.10. Technical success rate 95.08% → 92.44%.
> - **86% of the drop is a traffic-mix shift:** Delta Co-op Bank · UPI went from 2.9% to 14.4% of
>   volume (QuickTix: 390 → 10,871 attempts) while its own rate stayed flat (~69%). At last
>   week's mix the rate would be **87.18%**.
> - **One genuine degradation:** Coastal Bank · NetBanking 87.80% → 81.44% (z = −3.5), top failure
>   reason BANK_DOWNTIME.
> - **Caveats:** 411 duplicate rows on 20–22 Oct (de-duplicated); incomplete data on 14 Oct.
>
> Each finding links to the query that produced it (Q1, Q3, Q9 …).

Built for the *Enterprise AI Business Analyst* challenge. Data is synthetic, generated with
documented, planted scenarios so every answer can be checked against ground truth.

---

## Contents
[Quick start](#quick-start) · [Deploy](#deploy-to-streamlit-community-cloud) · [Using the app](#using-the-app) · [Roles](#roles-and-what-each-can-run) ·
[How it works](#how-it-works) · [Failure scenarios](#failure-and-exception-scenarios) ·
[Evaluation](#evaluation) · [Project structure](#project-structure) · [Configuration](#configuration) ·
[Troubleshooting](#troubleshooting) · [Assumptions and limitations](#assumptions-and-limitations)

---

## Quick start

Requires Python 3.11+.

**macOS / Linux**
```bash
git clone https://github.com/gitone912/payments-analyst && cd payments-analyst
pip install -r requirements.txt
python data/generate_data.py        # builds data/payments.duckdb (~285k transactions, seeded, ~5 s)
python run_app.py                   # opens http://localhost:8501
```

**Windows (cmd)**
```bat
git clone https://github.com/gitone912/payments-analyst && cd payments-analyst
python -m pip install -r requirements.txt
python data\generate_data.py
python run_app.py
```

Sign in with a demo account (one per role):

| Username | Password | Role |
|---|---|---|
| `head.payments` | `demo-head` | Head of Payments: all merchants, no PII |
| `risk.ops` | `demo-risk` | Risk Ops: all merchants, PII, full technical detail |
| `kam.a` | `demo-kam` | Key Account Manager: Portfolio A merchants only |

These are demo accounts (salted PBKDF2 hashes in `config/users.demo.json`); see
[Configuration](#configuration) for real users or single sign-on.

### Turn on the LLM planner (optional)

The app works fully without a model (deterministic planner). To use Gemini:

```bash
cp .env.example .env          # Windows: copy .env.example .env
# edit .env: LLM_PROVIDER=gemini and GEMINI_API_KEY=<your key>
python -m analyst.llm_check   # one real call per model; explains any failure in plain words
python run_app.py
```

`.env` is git-ignored. The sidebar shows **LLM on** with the model names when it works. Any
OpenAI-compatible endpoint (OpenAI, Groq, OpenRouter, Ollama, vLLM) also works:
`LLM_PROVIDER=openai`, `OPENAI_API_KEY`, optional `OPENAI_BASE_URL` and `LLM_MODEL`.

### Other entry points

```bash
pip install -r requirements-dev.txt                         # adds pytest
python -m pytest -q tests                                   # unit tests
python evals/run_evals.py --paraphrases                     # known-answer evals (rule planner)
python evals/run_evals.py --planner llm --paraphrases       # the same, against the live model
python cli.py "why did success rate drop last week?" --sql  # terminal client, prints executed SQL
python cli.py "why did success rate drop last week?" --then "now break that down by bank"
```

---

## Deploy to Streamlit Community Cloud

Free hosting straight from this GitHub repository (Vercel is not suitable: it runs short-lived
serverless functions, while a Streamlit app is a long-running server holding a WebSocket per user).

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. **Create app → Deploy a public app from GitHub**: repository `gitone912/payments-analyst`,
   branch `main`, main file `app.py`; choose the app URL.
3. **Advanced settings**: Python 3.12 or 3.13, and paste into **Secrets**:
   ```toml
   LLM_PROVIDER = "gemini"
   GEMINI_API_KEY = "your-key"
   LLM_MIN_INTERVAL_S = "4.5"   # free-tier keys allow 15 requests/minute
   ```
4. **Deploy.** The first start installs the pinned requirements, and the first page load builds the
   database (~10 s). Leave the secrets out to run without an LLM (rule planner only).

Notes: top-level secrets are copied into the environment at start-up (`app.py`), so the same
variables as `.env` apply. The app is public: anyone with the link can sign in with the demo accounts
and use your model quota, so use a dedicated key. Community Cloud apps sleep after inactivity and
wake on the next visit; the audit log resets when the app restarts.

---

## Using the app

**Ask** (main page). Type a question and press *Analyse*. Answers read like a short analyst report:

- **Headline** with a confidence badge, and a status line: planner used, queries run, time, model cost.
- **Summary**: Findings (each tagged with its query), Likely causes (hypotheses, never stated as
  fact), Recommended actions, Caveats, and Method (dataset, metric definition, period, comparison
  period, filters, assumptions, why this confidence).
- **Charts**: daily trend with data-quality days marked, mix-vs-rate contribution bars, breakdowns.
- **Evidence** (collapsed): every query that ran, the result tables and the full trace (steps,
  model calls, the JSON plan).

What it understands:

| You can ask | Example |
|---|---|
| Why something changed | "Why did success rate drop last week?", "What drove the rise in failure rate last week?" |
| Breakdowns and comparisons | "Success rate by bank last week", "Compare UPI vs Card success rate in October" |
| Trends | "Show daily success rate trend", "Success rate trend by method", "Success rate by week" |
| Specific periods | "Success rate on 14 Oct", "…from 1 Oct to 10 Oct", "…first week of October", "…yesterday", "…in the last 14 days" |
| Counts and money | "How many transactions failed yesterday?", "GMV by category last week" |
| Technical vs business | "Technical success rate by bank", "Business decline rate by method in October" |
| Source reconciliation | "Do settlements agree with transactions on success rate last week?" |
| Conditions (LLM writes guarded SQL) | "Share of transactions above 5000 INR last week", "Success rate on weekends last week" |
| Follow-ups | "Now break that down by bank", "What about UPI?" (**New conversation** clears context) |
| Several things at once | "Show the daily success rate trend and why did it drop last week" (one tab per part) |

Typos in known words are corrected and shown ("Read 'sucess' as 'success'"). When a question is
ambiguous, names something the data can't apply, or has no period for a "why" question, the app
asks instead of guessing.

**SQL console**: hand-written, read-only SQL through the same safety guard and access scope as
model-written SQL, with examples of unsafe queries it blocks. **Audit log**: your own queries
(who, role, when, outcome).

---

## Roles and what each can run

Defined once in `analyst/rbac.py`, **enforced by the engine and the SQL guard** (not only hidden in
the UI), and shown to each user under *Your access* in the sidebar.

| Capability | Head of Payments | Risk Ops | KAM (Portfolio A) |
|---|---|---|---|
| Ask questions | all merchants | all merchants | Portfolio A merchants only |
| Raw ingested data (`transactions_raw`, with duplicates) | yes | yes | no |
| Customer PII (`customer_vpa`) | no | yes | no |
| SQL console (hand-written, read-only, guarded) | yes | yes | no (engine refuses) |
| Executed SQL (physical tables, row-level scope rewrite) | no | yes | no (engine strips it) |

Out-of-scope requests are denied with the rule that applied (`pii`, `row_scope`, `table_scope`,
`sql_console`); merchant scope is also applied as a row-level rewrite on every query, as a second
layer. Suggested questions only include ones the signed-in role can run. The audit log always keeps
the executed SQL.

---

## How it works

```
question
  │  spelling correction · split multi-part questions · attach follow-up context
  ▼
Planner ─ LLM (JSON plan against the semantic catalog; backup model, retry-after, circuit breaker)
  │       └─ on outage / invalid output / rejected key → deterministic rule planner
  ▼
Plan checks ─ known metrics/dimensions · period inside the data · ambiguity → ask ·
  │           every condition in the question applied (else guarded custom SQL or ask) ·
  │           LLM dates inside the period asked · unknown words never silently ignored
  ▼
Compiler ─ plan → SQL over logical tables (one definition per metric in semantic.py)
  ▼
SQL guard (sqlglot AST) ─ single read-only SELECT · allow-listed functions only · catalog tables only ·
  │                       role checks (tables, PII, merchants) · join/scan limits · scope rewrite · LIMIT
  ▼
Executor ─ read-only DuckDB, one cursor per query, 8 s timeout, result cache, JSONL audit log
  ▼
Analysis ─ headline + 95% CI · technical vs business split · mix-vs-rate decomposition over
  │        7 segmentations · significance-filtered drivers · failure-reason drill-down
  ▼
Answer ─ findings [query ids] · likely causes · actions · caveats · method · confidence + reasons
```

### Key design decisions

- **The LLM plans; code computes.** The model outputs a JSON plan that references the catalog (or,
  only when the catalog can't express a condition, one SELECT). Numbers come from executed SQL and
  the text is built from those results, so the model cannot invent a figure.
- **Mix vs rate decomposition.** `ΔR = Σ Δwᵢ(r̄ᵢ − Rᴬ) + Σ Δrᵢ·w̄ᵢ`, exact and unit-tested. The primary
  view is the segmentation where mix explains most; rate drivers come only from that view, because
  at a coarser level a mix shift *inside* a segment looks like that segment getting worse.
- **Technical vs business first.** Failures are classified once (technical: TIMEOUT, BANK_DOWNTIME;
  business: insufficient funds, customer drop-off, risk and issuer policy). Every success-rate
  diagnosis splits the change exactly into technical, business and pending parts; the hypothesis
  and first action follow from which side moved.
- **Fail closed, and say so.** The guard allows only listed functions. A word the planner has no rule
  for gets a question, not an answer to a different question; an LLM plan must account for it or
  the answer is marked low confidence. LLM SQL must visibly apply each stated condition, its dates
  must sit inside the period asked, and the classic `created_at <= 'last day'` mistake (which drops
  that whole day) is corrected before execution and disclosed.
- **Clarify rather than guess** when the period or the measure is missing; lower-risk defaults
  (e.g. "last 7 days" for a snapshot) are listed as assumptions.
- **Honest uncertainty.** Confidence interval on the change, z-scores for drivers, data-quality caveats
  only when they overlap the period, a premise check ("you asked why it dropped; it rose").
- **Safe for concurrent users.** One engine serves all sessions; per-request state travels with the
  request; each query has its own cursor; a threaded test checks that contexts never mix.
- **Resilient to the model provider.** Backup models, short provider-requested waits, a circuit
  breaker and a time budget; after that the deterministic planner answers and the trace says why.

---

## Data

`data/generate_data.py` builds a seeded DuckDB database: 8 full weeks, **31 Aug – 25 Oct 2026**
("today" is the last day). 285,654 transaction rows as ingested (285,243 unique), 40 merchants,
6 banks, 4 payment methods, 2,240 settlement rows. All names are fictional.

| Table | Contents |
|---|---|
| `transactions` | one row per payment attempt (de-duplicated view): time, merchant, bank, method, amount, status (SUCCESS / FAILED / PENDING), failure reason, customer VPA (PII) |
| `transactions_raw` | the same, exactly as ingested (includes duplicates) |
| `merchants` | name, category, city tier, onboarding date, KAM portfolio |
| `banks`, `payment_methods` | dimensions |
| `settlements` | finance system's daily roll-up per merchant (resolves PENDING later) |

Planted scenarios (the ground truth the evals check):

| # | Scenario | Detail |
|---|---|---|
| 1 | Mix shift | From 19 Oct a QuickTix sale adds ~1,500 attempts/day, mostly UPI via Delta Co-op Bank (~70% success rate) |
| 2 | Genuine degradation | From 21 Oct, Coastal Bank NetBanking drops ~11 pts (BANK_DOWNTIME) |
| 3 | Duplicates | 20–22 Oct: an ingestion retry re-inserts 411 FAILED rows |
| 4 | Missing data | 14 Oct: only rows before 12:00 landed |
| 5 | Missing dimension | 8–10 Oct: some UPI rows have no bank_id |
| 6 | Source disagreement | settlements resolve most PENDING to SUCCESS, so they report ~1 pt more success |
| 7 | PII | customer VPA readable by Risk Ops only |

---

## Failure and exception scenarios

| Scenario | What happens | Try |
|---|---|---|
| Missing or duplicated records | Data-health panel flags them; analysis uses the de-duplicated view; caveats appear when the period overlaps; confidence drops | "Why did success rate drop last week?", "Success rate on 14 Oct", "…raw including duplicates" |
| Request outside access scope | Denied with the rule name; merchant scope also enforced as a row-level rewrite | Head of Payments: "Show customer VPAs with the most failures last week"; KAM: "Success rate for QuickTix last week" |
| Two sources disagree | Transactions vs settlements reconciled; the gap is explained (PENDING resolved later) with guidance on which to use | "Do settlements agree with transactions on success rate last week?" |
| Decline explained by mix shift | Exact mix-vs-rate split, like-for-like rate, contribution chart | "Why did success rate drop last week?" |
| Ambiguous period or denominator | Clarifying question with clickable options | "Why did success rate drop?", "How are we performing last week?" |
| Expensive or unsafe query | Blocked with the rule name; 8 s timeout | SQL console examples: DELETE, file read, cartesian join, raw-schema bypass |
| Model provider down / key rejected | Backup model, then the deterministic planner, flagged in caveats and trace | Sidebar: *Simulate LLM outage* |

### Requirements coverage

| Requirement | Where |
|---|---|
| ≥ 2 related datasets | 6 tables above |
| Natural-language questions, SQL generated and executed | Ask page; LLM or rule planner; compiler + guard + executor |
| Query and analysis steps shown | Evidence → Queries, Trace |
| Charts and tables | Charts tab; Evidence → Data |
| Dataset, filters and period cited | Method block; query id on every finding |
| Facts vs hypotheses vs recommendations | Findings / Likely causes / Recommended actions |
| Role-based access and query restrictions | Roles table above |
| Data-quality checks and warnings | Data health panel; caveats; confidence |
| Cohort comparison / drill-down | Segment decomposition, technical vs business, failure reasons, follow-ups |
| Cost/timeout controls and caching | Guard limits, 8 s timeout, result and plan caches |
| Known-answer evaluation | [Evaluation](#evaluation) |
| Confidence and uncertainty | Confidence badge + reasons, CI, z-scores |

---

## Evaluation

`evals/run_evals.py` asks the engine known questions and compares the answers with **independent,
hand-written SQL on the raw tables** (not the engine's own compiler). It covers headline numbers,
the planted drivers, data-quality caveats, clarifications, access denials, guard blocks, LLM-written
SQL for conditions, follow-ups, multi-part questions and the provider-outage fallback. A held-out set
of reworded questions (never used as prompt examples) checks that planning generalises.

The table below is generated from saved runs; it is never typed by hand.

<!-- eval-results:start -->
<!-- generated by evals/render_results.py from evals/results/*.json; do not edit -->
| Planner | Core checks passed | Wrong answers | Provider errors | Held-out paraphrases | Run (UTC) · commit |
|---|---|---|---|---|---|
| Deterministic rules | 41/41 | 0 | 0 | 9/10 | 2026-11-21 17:21 · `93d7c70` |
| LLM (live) `gemini-3.5-flash-lite` | 41/41 | 0 | 0 | 10/10 | 2026-11-21 17:29 · `fc7458f` |

- Deterministic rules paraphrases not passed: "How did Card do compared with NetBanking in September?" (FAIL)
- LLM (live): models tried in order `gemini-3.5-flash-lite` → `gemini-3.1-flash-lite`; answers came from `gemini-3.1-flash-lite` ×4, `gemini-3.5-flash-lite` ×29.
<!-- eval-results:end -->

```bash
python evals/run_evals.py [--planner llm] --paraphrases --save   # writes evals/results/<planner>.json
python evals/render_results.py                                    # rebuilds the table above (--check for CI)
```

Exit codes: `0` all passed, `1` at least one wrong answer, `3` only model-provider errors remain.
The eval runner never retries: provider resilience is the app's job, so the numbers show what a user
gets. Unit tests (`tests/test_core.py`) cover the decomposition maths, LLM retry/validation, the
circuit breaker, guard allow-list, auth, role enforcement, concurrency and every bug found in testing.

---

## Project structure

```
payments-analyst/
├── app.py                  Streamlit entry: settings, engine, sign-in, navigation
├── run_app.py              Launcher: clean Ctrl+C, no Windows connection-reset noise
├── cli.py                  Terminal client (--sql, --then for follow-ups, --role, --planner)
├── analyst/                The engine (no UI code)
│   ├── engine.py           Orchestration: plan checks, analysis, answer composition
│   ├── semantic.py         Metric and dimension catalog; failure taxonomy
│   ├── rule_planner.py     Deterministic planner: dates, metrics, spelling, follow-ups, multi-part
│   ├── llm.py              LLM planner: providers, backup models, circuit breaker, plan validation
│   ├── plan.py             Plan and Window types, validation
│   ├── compiler.py         Plan → SQL
│   ├── guard.py            SQL safety and access control (sqlglot AST)
│   ├── sqlfix.py           Disclosed fix for LLM SQL that drops the last day
│   ├── executor.py         Read-only DuckDB, timeout, cache, audit log
│   ├── drivers.py          Mix-vs-rate decomposition, confidence interval, z-scores
│   ├── dq.py               Data-quality checks
│   ├── rbac.py             Roles and capabilities
│   ├── auth.py             Sign-in (password file or SSO header); `check` command
│   ├── config.py           .env loader
│   ├── llm_check.py        `python -m analyst.llm_check`
│   └── charts.py           Plotly figures
├── ui/                     Presentation only
│   ├── pages.py            Sign-in, Ask, SQL console, Audit log, sidebar
│   ├── answer.py           Report-style answer view
│   ├── components.py       Small rendering helpers
│   └── theme.py            Stylesheet for the app's own elements
├── .streamlit/config.toml  Design system: Inter, one accent, light and dark palettes
├── data/generate_data.py   Synthetic data with planted scenarios
├── evals/                  Known-answer evals, saved results, README table generator
├── tests/test_core.py      Unit tests
└── config/users.demo.json  Demo accounts
```

---

## Configuration

Set in `.env` (see `.env.example`) or the environment; values already set in the terminal win.

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `none` | `gemini` or `openai` (any OpenAI-compatible API) |
| `GEMINI_API_KEY` / `OPENAI_API_KEY` | | API key (Gemini key is sent in a header, never the URL) |
| `OPENAI_BASE_URL` | OpenAI | Groq, OpenRouter, Ollama, vLLM … |
| `LLM_MODEL` | `gemini-3.5-flash-lite` | Primary model |
| `LLM_FALLBACK_MODELS` | `gemini-3.1-flash-lite` | Backup models, comma-separated |
| `LLM_MIN_INTERVAL_S` | `0` | Pace requests on a free-tier key (15 requests/minute → `4.5`) |
| `LLM_MAX_RETRY_WAIT_S` / `LLM_REQUEST_BUDGET_S` | `5` / `20` | Longest provider-requested wait honoured; time budget per question |
| `LLM_BREAKER_THRESHOLD` / `LLM_BREAKER_COOLDOWN_S` | `3` / `60` | Skip a model after N failures in a row, for N seconds |
| `AUTH_MODE` | `password` | `header`: trust an SSO proxy's identity header (only behind that proxy) |
| `AUTH_HEADER` | `X-Forwarded-Email` | Identity header in `header` mode |
| `USERS_FILE` | `config/users.demo.json` | Accounts and roles; add users with `python -m analyst.auth add <user> <role>` |
| `AUDIT_LOG` | `logs/audit.jsonl` | Audit log path (empty to disable) |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `'streamlit' is not recognized` (Windows) | Use `python run_app.py` or `python -m streamlit run app.py` |
| Sidebar says **LLM off** / answers show "Model cost: none" | Create `.env` from `.env.example`, add the key, restart |
| "API key rejected" caveat | `python -m analyst.llm_check` gives the reason; check the key was copied completely |
| Sign-in fails for every account | `python -m analyst.auth check`; if the browser autofilled the fields, type them instead |
| Ctrl+C hangs on "Stopping…" | Start with `python run_app.py`, which always stops within 5 s |
| Answers look stale after `git pull` | Stop and restart the app |
| Deployed app says **LLM off** | Add `LLM_PROVIDER` and `GEMINI_API_KEY` under the app's *Settings → Secrets*, then reboot the app |

---

## Assumptions and limitations

- **Synthetic data with planted scenarios.** The independent ground-truth SQL and held-out reworded
  questions reduce, but don't remove, the bias of grading against scenarios we designed.
- **"Today" is the last day in the data** (25 Oct 2026); "last week" is 19–25 Oct vs 12–18 Oct.
- **Success rate** = SUCCESS / all attempts (PENDING in the denominator); the excluding-pending and
  technical variants are available and named on every answer.
- **Driver attribution is descriptive, not causal**; causes are always shown as hypotheses.
- **Authentication** is a local users file or a trusted SSO header; there is no built-in OIDC flow.
- **Cost controls** are AST rules, limits and a timeout, not a cost estimate before execution; caches
  are in memory.
- **The rule planner's vocabulary is deliberately strict**: unusual phrasing gets a clarifying question
  (the LLM planner handles it). Extend `_KNOWN_WORDS` in `rule_planner.py` as needed.
- **The failure taxonomy** is a fixed mapping of this dataset's reasons; real issuer/NPCI response
  codes would need a maintained mapping.
- **LLM cost** uses approximate list prices in `llm.py`.
