"""Data models for the Budgeteer SDK.

Defines all core entities: budget accounts, run budgets, step context/decisions,
telemetry records, and standardized LLM/tool response types.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class BudgetScope(str, Enum):
    """Scope level for a budget account."""

    USER = "user"
    ORG = "org"
    PROJECT = "project"


@dataclass
class BudgetAccount:
    """Spending limits for a scope (user, org, or project)."""

    scope: BudgetScope
    scope_id: str
    limit_usd_per_day: float | None = None
    limit_tokens_per_day: int | None = None
    limit_runs_per_day: int | None = None


@dataclass
class RunBudget:
    """Hard caps for a single agent run."""

    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    hard_usd_cap: float | None = None
    hard_token_cap: int | None = None
    hard_latency_cap_ms: float | None = None
    max_tool_calls: int | None = None


@dataclass
class ModelTier:
    """A model option with its cost parameters."""

    name: str
    cost_per_prompt_token: float
    cost_per_completion_token: float
    max_context_window: int
    tier: str = "standard"


@dataclass
class StepContext:
    """Input context provided to the policy engine before each agent step."""

    run_id: str
    step_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_type: str | None = None
    messages: list[dict[str, Any]] | None = None
    tools_available: list[str] | None = None
    budget_account: BudgetAccount | None = None
    run_budget: RunBudget | None = None
    current_run_cost_usd: float = 0.0
    current_run_tokens: int = 0
    current_run_tool_calls: int = 0
    elapsed_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StepDecision:
    """Output of the policy engine: control actions for a step."""

    model: str
    max_tokens: int = 1024
    temperature: float = 0.7
    context_window: int = 8192
    tool_calls_allowed: int = 5
    retrieval_enabled: bool = True
    retrieval_top_k: int = 3
    degrade_level: int = 0
    degrade_reason: str | None = None


@dataclass
class StepMetrics:
    """Observed metrics from executing a step."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    tool_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    tool_calls_made: int = 0
    success: bool | None = None


@dataclass
class RunRecord:
    """Summary record for a complete agent run."""

    run_id: str
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    total_cost_usd: float = 0.0
    total_tokens: int = 0
    total_latency_ms: float = 0.0
    total_steps: int = 0
    total_tool_calls: int = 0
    success: bool | None = None


@dataclass
class StepRecord:
    """Record of a single step's decision and outcome."""

    run_id: str
    step_id: str
    decision: StepDecision
    predicted: StepMetrics | None = None
    actual: StepMetrics | None = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class ToolRecord:
    """Record of a single tool execution."""

    run_id: str
    step_id: str
    tool_name: str
    duration_ms: float
    tokens_used: int = 0
    success: bool = True
    error: str | None = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class LLMResponse:
    """Standardized response from an LLM call."""

    content: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    raw_response: Any = None


def compute_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    model_tiers: list[ModelTier],
) -> float:
    """Compute cost in USD for a given model and token counts.

    Looks up the model in *model_tiers* and multiplies by per-token rates.
    Returns 0.0 if the model is not found.
    """
    for tier in model_tiers:
        if tier.name == model:
            return (
                prompt_tokens * tier.cost_per_prompt_token
                + completion_tokens * tier.cost_per_completion_token
            )
    return 0.0


@dataclass
class ToolResult:
    """Standardized result from a tool execution."""

    tool_name: str
    output: Any = None
    success: bool = True
    error: str | None = None
    duration_ms: float = 0.0
    tokens_used: int = 0


@dataclass
class StepResult:
    """Result of an orchestrated step execution."""

    decision: StepDecision
    llm_response: LLMResponse
    tool_results: list[ToolResult] = field(default_factory=list)
    metrics: StepMetrics = field(default_factory=StepMetrics)
    context_was_truncated: bool = False
    roi_decisions: dict[str, Any] | None = None
    suggested_question: Any | None = None  # ClarifyingQuestion when available
