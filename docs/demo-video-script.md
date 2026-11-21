# Demo video: description and script

Target length: about 4 minutes. Record at 1440×900 or larger, browser zoom 100%, light or dark mode.
Before recording: `python run_app.py`, LLM on (`python -m analyst.llm_check` says WORKING), planner on
**Auto**, and sign out so the video starts on the sign-in page.

---

## 1. Video description (paste into YouTube / LinkedIn / the submission form)

**Title:** Payments AI Business Analyst: answers you can check, not just answers

**Description:**

Leaders ask "why did our payment success rate drop?" and analysts spend hours finding the data and
writing the explanation. This is an AI business analyst that answers in seconds, and shows its work.

Ask a question in plain English. The system plans the analysis, validates and runs the SQL, splits the
change by segment, and returns a short report: the findings (each linked to the query behind it),
likely causes, recommended actions, caveats and a confidence level. The language model only decides
what to compute; every number comes from an executed, guarded query, so the AI cannot invent a figure.

What the demo shows:
- "Why did success rate drop last week?": −2.59 pts, of which 86% is a traffic-mix shift (a merchant
  sale pushed volume onto a weak bank route), plus one genuine bank outage, and a technical-vs-business
  split of the failures
- Evidence for every claim: the queries, result tables and the full trace
- Follow-up questions, typo correction, multi-part questions and clarifying questions when a question is ambiguous
- Data-quality caveats (duplicate and missing records) and a transactions-vs-settlements reconciliation
- Role-based access: three roles with different data, PII and SQL permissions, enforced by the engine
- A SQL safety guard that blocks writes, file access and expensive queries
- Graceful fallback when the language model is unavailable
- 41 known-answer evaluation checks against independent SQL, with the live LLM

Built with Python, DuckDB, sqlglot, Streamlit, Plotly and Gemini (any OpenAI-compatible model works).
Data is synthetic, with documented planted scenarios so every answer can be verified.

Code: https://github.com/gitone912/payments-analyst

**Chapters:**
```
0:00 The problem
0:20 Sign in and roles
0:35 Why did success rate drop? The answer
1:20 Evidence: charts, queries, trace
1:50 Follow-ups, typos and multi-part questions
2:20 Ambiguity, data quality and source disagreement
2:50 Access control across three roles
3:25 SQL safety guard
3:40 When the AI model is down
3:55 Evaluation and wrap-up
```

---

## 2. Script

Each scene: what to do on screen, then what to say. Spoken text is about 130 words per minute.

### 0:00 The problem (title card or the sign-in page)

> Payment leaders ask questions like "why did our success rate drop last week, and what should we do?"
> Today an analyst spends hours finding the data and writing the answer. This is an AI business analyst
> that answers in seconds, and, more importantly, shows exactly how it got there.

### 0:20 Sign in and roles

**Do:** sign in as `head.payments` / `demo-head`. Point at the sidebar: *Your access*.

> Every user signs in, and their role decides what they can see. I'm the Head of Payments: all
> merchants, but no customer personal data, and no access to the internal SQL. We'll switch roles later.

### 0:35 The main question

**Do:** click the suggestion **Why did success rate drop last week?** Let the answer load. Scroll slowly
through Findings, Likely causes and Recommended actions.

> Success rate fell from 87.5 to 84.9 percent, a drop of 2.6 points, and the confidence interval tells us
> it's real, not noise.
>
> First, it's technical, not customers: technical failures account for 2.3 of the 2.6 points.
>
> Second, and this is the key insight, 86 percent of the drop is a change in traffic mix, not payments
> getting worse. A QuickTix sale pushed a lot of UPI volume through Delta Co-op Bank, a route that was
> always weaker. At last week's mix, the success rate would be 87.2 percent.
>
> Third, one thing did genuinely break: Coastal Bank NetBanking fell six points, and the failures are
> bank downtime.
>
> Notice the structure: findings are facts from the data, likely causes are clearly marked as
> hypotheses, and the recommendations follow from them: fix routing for the weak route, confirm the
> campaign with the account team, open an incident with Coastal Bank.

### 1:20 Evidence

**Do:** open **Charts** (trend with red data-quality markers, mix-vs-rate bars). Back to **Summary**, scroll
to **Method**, then open **Evidence → Queries → Q1**, then **Trace**.

> The charts show the daily trend, with data-quality problems marked, and which segments moved the
> number, split into mix and rate.
>
> Every finding has a query reference. Here's the method: the dataset, the exact metric definition, the
> period and the comparison period. And under Evidence, every query that ran, each one checked by the
> SQL safety guard first, plus the full trace of how the answer was built. Nothing here is the model's
> opinion; every number comes from a query you can read.

### 1:50 Follow-ups, typos, multi-part

**Do:** type **now break that down by bank** → Analyse. Then **New conversation**. Then type
**show daily sucess rate trend and why did it drop last week** (with the typo) → Analyse; show the two
part tabs and the "Read 'sucess' as 'success'" note.

> It keeps context, so I can just say "now break that down by bank". New conversation clears it.
>
> It fixes typos, and tells me it did. And if I ask two things at once, it answers each part
> separately.

### 2:20 Ambiguity, data quality, sources

**Do:** ask **Why did success rate drop?** → show the clarifying options, click **Last week vs the week
before**. Then ask **Do settlements agree with transactions on success rate last week?** Point at the
data-health panel in the sidebar.

> When a question is ambiguous, here there's no time period, it asks instead of guessing.
>
> The data has real problems, and it doesn't hide them: duplicate records from an ingestion retry, and a
> day with missing data. It removes the duplicates, adds caveats whenever they affect the answer, and
> lowers its confidence.
>
> And when two sources disagree, transactions say 84.9 percent and finance's settlements say 85.9,
> it explains why: pending payments that settle later.

### 2:50 Access control

**Do:** ask **Show customer VPAs with the most failures last week** → *Access denied*. Sign out, sign in as
`kam.a` / `demo-kam`: point at *Your access* (no SQL console in the menu), ask **Success rate for QuickTix
last week** → denied. Sign out, sign in as `risk.ops` / `demo-risk`, ask the VPA question again → answered;
open Evidence → Queries → a query → **Executed SQL**.

> Customer data is personal data, so the Head of Payments is refused, with the reason.
>
> A key account manager only sees their own merchants. QuickTix isn't theirs, so it's refused, and
> they don't even get the SQL console.
>
> Risk Ops can see customer data for investigations, and only they can see the executed SQL, including
> the row-level filter the system adds to every query. These rules are enforced in the engine, not just
> hidden in the interface.

### 3:25 SQL safety guard

**Do:** **SQL console** → run *Safe aggregate* (passes), then *Unsafe: DELETE* and *Unsafe: file read*
(blocked).

> Any SQL, whether the model wrote it or a person typed it, goes through the same guard: read-only,
> an allow-list of safe functions, no file access, no runaway joins, and an 8-second timeout.

### 3:40 When the model is down

**Do:** Ask page, turn on **Simulate LLM outage**, ask **Why did failure rate go up last week?** → answered;
show the caveat and the Trace step "fallback".

> If the AI provider is down, it tries a backup model, and then falls back to a deterministic planner.
> The answer still arrives, and it tells you it fell back.

### 3:55 Evaluation and wrap-up

**Do:** show the README's evaluation table on GitHub.

> To trust an AI analyst you have to test it. 41 known-answer checks compare its answers with
> independent, hand-written SQL, including the planted traps like the mix shift, and it passes all of
> them with the live model.
>
> So: answers in seconds, every number backed by a query, honest about uncertainty and data problems,
> and safe by design. Thanks for watching.

---

## 3. Recording checklist

- [ ] `git pull` on `main`, then `python run_app.py`
- [ ] LLM check says WORKING (otherwise the demo still works; the status line will say "Rules")
- [ ] Planner on **Auto**, outage toggle **off** at the start
- [ ] Close other tabs and notifications; zoom 100%
- [ ] Do one dry run: the first LLM call of a session is slower
- [ ] Never show `.env` or a terminal with the API key on screen
