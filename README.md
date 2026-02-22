# agent-budgeteer

Budget-aware controller for AI agents. Control cost, latency, and token usage with hard caps, graceful degradation, and intelligent routing.

## Features

- **Run budgets** — hard caps on USD, tokens, latency, and tool calls per run
- **Daily budget accounts** — per-user/org/project spending limits
- **Strategy routing** — automatic model selection across configured tiers
- **Graceful degradation** — 5-level ladder (full capabilities to strict mode)
- **Context management** — truncation, summarization, and retrieval packing
- **ROI heuristics** — value-of-information scoring for tools, retrieval, and clarifying questions
- **Calibration** — EMA-based correction factors that improve cost/latency predictions over time
- **Reporting** — run summaries, per-model stats, prediction accuracy (MAE/bias/MAPE), budget compliance

## Installation

```bash
pip install .
```

For YAML config file support:

```bash
pip install ".[yaml]"
```

## Quick Start

```python
from budgeteer.config import BudgeteerConfig
from budgeteer.models import RunBudget, StepContext, StepMetrics
from budgeteer.sdk import Budgeteer

# 1. Configure
config = BudgeteerConfig(default_model="gpt-4o-mini")
sdk = Budgeteer(config)

# 2. Start a run with a budget
run = sdk.start_run(run_budget=RunBudget(hard_usd_cap=1.0, hard_token_cap=50000))

# 3. Before each step — get control decisions
ctx = StepContext(run_id=run.run_id)
decision = sdk.before_step(ctx)
# Use decision.model, decision.max_tokens, decision.temperature for your LLM call

# 4. After each step — report actual usage
metrics = StepMetrics(prompt_tokens=120, completion_tokens=50, cost_usd=0.002, latency_ms=350)
sdk.after_step(ctx, decision, metrics)

# 5. End the run
sdk.end_run(run.run_id, success=True)
sdk.close()
```

## Configuration

Load from a JSON file:

```python
config = BudgeteerConfig.from_file("budgeteer.json")
```

Example `budgeteer.json`:

```json
{
  "default_model": "gpt-4o-mini",
  "default_max_tokens": 1024,
  "default_temperature": 0.7,
  "model_tiers": [
    {
      "name": "gpt-4o-mini",
      "cost_per_prompt_token": 0.00015,
      "cost_per_completion_token": 0.0006,
      "max_context_window": 128000,
      "tier": "cheap"
    },
    {
      "name": "gpt-4o",
      "cost_per_prompt_token": 0.005,
      "cost_per_completion_token": 0.015,
      "max_context_window": 128000,
      "tier": "premium"
    }
  ],
  "default_run_budget": {
    "hard_usd_cap": 1.0,
    "hard_token_cap": 50000
  }
}
```

When `model_tiers` are configured, the strategy router automatically selects the best model and degradation level for each step based on remaining budget.

## Architecture

```
Budgeteer (sdk.py)
  |
  +-- PolicyEngine (policy.py)
  |     |
  |     +-- StrategyRouter (router.py)    # model selection + degradation
  |     +-- Calibrator (calibrator.py)    # prediction correction factors
  |
  +-- TelemetryStore (telemetry.py)       # SQLite persistence
  +-- ContextManager (context_manager.py) # truncation + summarization
  +-- ROIEvaluator (roi.py)              # value-of-information heuristics
  +-- Reporter (reporting.py)            # aggregate reports
```

The SDK wraps the `before_step` / `after_step` lifecycle. Before each step, the policy engine evaluates budget constraints and returns a `StepDecision` with the model, token limits, and degradation level. After each step, actual metrics are recorded for calibration and reporting.

## Modules

| Module | Description |
|--------|-------------|
| `sdk.py` | Main entry point — run management and step lifecycle |
| `config.py` | Configuration loading from JSON/YAML |
| `models.py` | Core dataclasses (budgets, contexts, decisions, records) |
| `policy.py` | Budget enforcement and degradation logic |
| `router.py` | Strategy candidate generation, forecasting, and selection |
| `telemetry.py` | SQLite-backed storage for runs, steps, and tool calls |
| `context_manager.py` | Message truncation, summarization, and retrieval packing |
| `roi.py` | ROI/VOI heuristics for tools, retrieval, and clarifying questions |
| `calibrator.py` | EMA-based prediction correction factors |
| `reporting.py` | Run summaries, model stats, prediction accuracy, compliance |
| `llm_client.py` | LLM callable wrapper with usage tracking |
| `tool_executor.py` | Tool registry and execution with metric capture |
| `exceptions.py` | `BudgetExceededError` |

## Development

```bash
uv pip install -e ".[dev]"
pytest
```

## License

MIT
