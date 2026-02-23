"""Agent Budgeteer — control your agent's cost, latency, and tokens."""

from budgeteer.async_sdk import AsyncBudgeteer
from budgeteer.calibrator import Calibrator, CorrectionFactors, PredictionError
from budgeteer.config import BudgeteerConfig
from budgeteer.context_manager import ContextManager, RetrievalResult
from budgeteer.events import (
    BUDGET_EXCEEDED,
    BUDGET_WARNING,
    FALLBACK_TRIGGERED,
    RETRY_ATTEMPT,
    ROI_GATE_BLOCKED,
    RUN_ENDED,
    RUN_STARTED,
    STEP_COMPLETED,
    STEP_DECIDED,
    BudgeteerEvent,
    EventHook,
)
from budgeteer.exceptions import BudgetExceededError, RetryExhaustedError
from budgeteer.llm_client import LLMClient
from budgeteer.models import (
    BudgetAccount,
    BudgetScope,
    LLMResponse,
    ModelTier,
    RunBudget,
    RunRecord,
    StepContext,
    StepDecision,
    StepMetrics,
    StepRecord,
    StepResult,
    ToolRecord,
    ToolResult,
    compute_cost,
)
from budgeteer.policy import PolicyEngine
from budgeteer.reporting import Reporter
from budgeteer.roi import ClarifyingQuestion, ROIEvaluator, ROIResult, ROISignals
from budgeteer.router import CandidateStrategy, StrategyRouter
from budgeteer.sdk import Budgeteer
from budgeteer.telemetry import TelemetryStore
from budgeteer.tool_executor import ToolExecutor

__version__ = "0.1.0"

__all__ = [
    "AsyncBudgeteer",
    "Budgeteer",
    "BudgeteerConfig",
    "BudgeteerEvent",
    "BUDGET_EXCEEDED",
    "BUDGET_WARNING",
    "BudgetAccount",
    "Calibrator",
    "CorrectionFactors",
    "BudgetExceededError",
    "RetryExhaustedError",
    "BudgetScope",
    "CandidateStrategy",
    "EventHook",
    "FALLBACK_TRIGGERED",
    "ClarifyingQuestion",
    "ContextManager",
    "LLMClient",
    "LLMResponse",
    "ModelTier",
    "PolicyEngine",
    "PredictionError",
    "Reporter",
    "RETRY_ATTEMPT",
    "ROI_GATE_BLOCKED",
    "RUN_ENDED",
    "RUN_STARTED",
    "RetrievalResult",
    "ROIEvaluator",
    "ROIResult",
    "ROISignals",
    "RunBudget",
    "RunRecord",
    "STEP_COMPLETED",
    "STEP_DECIDED",
    "StepContext",
    "StepDecision",
    "StepMetrics",
    "StepRecord",
    "StepResult",
    "StrategyRouter",
    "TelemetryStore",
    "ToolExecutor",
    "ToolRecord",
    "ToolResult",
    "compute_cost",
]
