# Agent Budgeteer — MVP Project Plan

> **Status:** Draft — Pending Review
> **Version:** 0.1
> **Date:** 2025-02-22

---

## 1. Product Definition

### Goal

Ensure an agent meets user success targets while staying within:

- **Cost budgets** — per run / per user / per day
- **Token budgets** — prompt + completion, tool tokens, retrieval tokens
- **Latency budgets** — p95 per step, end-to-end SLA

### Non-Goals (Initially)

- Perfect optimal control (start with heuristics + calibration)
- Model-provider specific features beyond "call model / get usage"
- Fully autonomous learning in production without guardrails

### Target Environments

- Chat agents (interactive)
- Batch jobs (offline)
- Multi-tool agents (retrieval, code exec, web, DB)

---

## 2. Core User Stories

### Budget Enforcement

> "This user gets $X/day, this run gets $Y, this org gets $Z."

If budget is near exhaustion, the system **degrades gracefully** instead of failing abruptly.

### Dynamic Routing

For each step: choose model, max_tokens, temperature, context size, and tool strategy.

### Budget-Aware Planning

Before doing 5 tool calls, the agent learns to:

- Ask one decisive clarifying question
- Skip retrieval if low ROI
- Prefer cheaper tool/model combos for similar expected outcomes

### Telemetry + Outcomes

- **Track:** tokens, cost, latency, tool calls, failures, user satisfaction proxy / task success labels
- **Show:** "we saved 38% cost at same success rate" style reporting

---

## 3. High-Level Architecture

### Components

| Component | Inputs | Outputs |
|---|---|---|
| **Policy Engine** | user/org/run context, current spend, predicted spend, SLA constraints, task type | Control actions (model choice, max tokens, tool call limits, compression levels) |
| **Forecaster** | Candidate strategies, historical data | Predicted tokens/cost/latency per candidate; calibrated over time |
| **Planner Controller** | Budget state, task context | Budget-aware plan step: ask clarifying question, choose cheaper variant, limit tool calls |
| **Context Manager** | Messages, retrieval results | Truncated/summarized context within budget |
| **Tool ROI Evaluator** | Remaining budget, tool metadata | Go/no-go decision per tool call using VOI heuristics |
| **Telemetry + Storage** | All step/run data | Structured JSON events (`run_id`, `step_id`, model, tokens, cost, latency, decision reason) |

### Deployment Shapes

- **Library mode (MVP):** wraps LLM calls and tool calls inside your code
- **Sidecar service (later):** local/cluster service receiving step requests, returning policy decisions

---

## 4. Data Model

### Budget Entities

```
BudgetAccount {
  scope: user | org | project
  limit_usd_per_day: number
  limit_tokens_per_day: number
  limit_runs_per_day: number
}

RunBudget {
  run_id: string
  hard_usd_cap: number
  hard_token_cap: number
  hard_latency_cap_ms: number
}

Policy: rules + fallback ladders
```

### Telemetry Entities

```
RunRecord   { start/end time, total cost/tokens/latency, success label }
StepRecord  { input features, chosen strategy, predicted vs actual metrics, outcomes }
ToolRecord  { tool name, duration, bytes/tokens, success/failure }
```

---

## 5. Control Loop

At each agent step:

1. **Build candidate strategies** (5-12 options)
   - `(model A, context 8k, retrieval off)`
   - `(model B, context 32k, retrieval top-3)`
   - `(model B, context 16k, retrieval top-1, ask clarifying question)`
   - `(model C cheap, context 8k, strict output, no tools)`

2. **Forecast per candidate**
   - Expected prompt tokens, completion tokens, tool tokens
   - Expected cost
   - Expected latency
   - Rough expected success probability (initially heuristic)

3. **Apply policy constraints**
   - Remove candidates that violate hard caps
   - If none remain: pick best "degrade ladder" option

4. **Choose** — pick candidate with best expected utility under budgets

5. **Execute** — make model call and tool calls within imposed limits

6. **Observe + log** — actual usage metrics, step outcome

7. **Update calibrations**

---

## 6. Algorithms

### A) Token + Cost Prediction

**MVP (deterministic estimator):**

```
prompt_tokens     ~ tokenize_len(messages + tools + retrieved_snippets)
completion_tokens ~ min(max_tokens, heuristic(task_type))
```

**Calibration layer:**
- Per-model correction factors (moving average error)
- Per-task-type completion distributions

### B) Latency Prediction

Rolling p50/p95 per model, tool, and context-size bucket.

```
E2E latency ~ model_latency + sum(tool_latencies) + overhead
```

### C) Value-of-Information / ROI (Heuristic)

```
ROI = benefit_score / (predicted_cost + lambda * predicted_latency)
```

**Benefit signals:** question ambiguity, need for fresh facts, previous failures
**Cost signals:** remaining budget, predicted tokens/latency

### D) Budget-Aware Clarifying Question

If ambiguity is high and ROI of asking is high, ask a single targeted question from a template library:

- "What output format do you need (JSON/table/text)?"
- "Is accuracy or speed more important?"
- "Do you want me to use web/retrieval or only provided data?"

---

## 7. Graceful Degradation Ladder

Ordered fallback set (policy-configurable, logged with reason):

| Level | Strategy |
|---|---|
| 1 | Full plan: strong model + retrieval + tools |
| 2 | Same model, reduced retrieval (top-k down, snippet size down) |
| 3 | Smaller model, minimal retrieval |
| 4 | No retrieval, internal reasoning + ask clarifying question |
| 5 | Strict summarization mode ("best effort answer, cite uncertainty, propose next step") |

---

## 8. MVP Scope

### MVP Deliverables

| # | Deliverable | Description |
|---|---|---|
| 1 | **Configurable Budgets** | YAML/JSON policies: per-user/org/run budgets, tool limits, degrade ladder |
| 2 | **Model Router** | Choose among 2-3 models (or tiers) via an abstraction |
| 3 | **Context Manager** | Truncation + summarization compression |
| 4 | **Tool Call Governor** | Max tool calls per run/step; block low-ROI calls when budget is tight |
| 5 | **Telemetry** | Persistent logs + simple report: spend/tokens/latency vs success |

### MVP Success Criteria

- [ ] Cost reduced at similar task success rate (demonstrated on benchmark suite)
- [ ] No budget violations across test runs
- [ ] Predictor error decreases over time (calibration works)

---

## 9. Evaluation Plan

### Metrics

| Category | Metrics |
|---|---|
| **Budget compliance** | % runs within caps |
| **Efficiency** | cost/run, tokens/run, tool calls/run |
| **Latency** | p50/p95 end-to-end |
| **Quality** | task success (binary/graded), structured output validity (JSON parse rate), user satisfaction proxy (thumbs up/down, "follow-up needed" rate) |

### Benchmark Harness

- 20-50 representative tasks (domain-specific)
- Deterministic tool stubs or cached retrieval results for repeatability
- A/B comparison: baseline agent vs agent + Budgeteer (different policies)

---

## 10. Implementation Milestones

### Milestone 1 — Skeleton + Interfaces

- [ ] Define Budgeteer SDK API: `before_step(context) -> decision`, `after_step(observed_metrics) -> update`
- [ ] Implement `LLMClient` wrapper capturing usage
- [ ] Implement `ToolExecutor` wrapper capturing duration/results
- [ ] Basic logging to file/DB

### Milestone 2 — Policy Engine + Hard Caps

- [ ] Budget accounting: per-run caps
- [ ] Budget accounting: per-user per-day caps (simple KV store)
- [ ] Enforce max tool calls
- [ ] Enforce max tokens
- [ ] Enforce max latency (stop early + respond with partial)

### Milestone 3 — Strategy Router + Degradation Ladder

- [ ] Candidate generation
- [ ] Pick strategy under constraints
- [ ] Implement degrade ladder rules

### Milestone 4 — Context Manager v1

- [ ] Truncation rules
- [ ] Summarize older messages (cheap model) when context must shrink
- [ ] Retrieval packing logic (if retrieval exists)

### Milestone 5 — ROI Heuristics + Clarifying Question Mode

- [ ] Implement VOI heuristics
- [ ] Add decisive-question selection templates
- [ ] Add "skip retrieval if low ROI" guard

### Milestone 6 — Calibration + Reporting

- [ ] Compare predicted vs actual
- [ ] Update correction factors
- [ ] Produce reports + dashboards (optional)

---

## 11. Tech Choices

| Decision | MVP | Later |
|---|---|---|
| **Language** | Python (fast iteration) or TypeScript (agent ecosystem) | — |
| **Storage** | SQLite / Postgres | Redis for budget counters + Postgres for telemetry |
| **Observability** | OpenTelemetry events | Prometheus metrics + Grafana dashboard |

---

## 12. Risks + Mitigations

| Risk | Mitigation |
|---|---|
| Prediction error causes bad decisions | Conservative bounds + fallback ladder; calibrate continuously |
| Quality regressions from over-degrading | Minimum quality floor policies; task-type exemptions |
| Tool ROI too heuristic | Start heuristic, then learn lightweight classifier/regressor from telemetry |
| Logging sensitive data | Redact prompts; store hashes; configurable PII filters |

---

## 13. Post-MVP Extensions (V1+)

These are **out of scope** for MVP but documented for roadmap planning:

- Learned success predictor (features -> probability of success per strategy)
- Per-user personalization ("this user tolerates slower but accurate")
- Auto-policy tuning (Bayesian optimization / bandits) with guardrails
- Multi-objective optimization (Pareto frontier of cost/latency/quality)
- SLA-aware scheduling in multi-agent systems

---

