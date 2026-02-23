"""Event system for Budgeteer observability.

Defines event types and a hook mechanism for external integrations
(dashboards, alerting, custom telemetry).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

# Event type constants
RUN_STARTED = "run_started"
RUN_ENDED = "run_ended"
STEP_DECIDED = "step_decided"
STEP_COMPLETED = "step_completed"
BUDGET_WARNING = "budget_warning"
BUDGET_EXCEEDED = "budget_exceeded"
RETRY_ATTEMPT = "retry_attempt"
FALLBACK_TRIGGERED = "fallback_triggered"
ROI_GATE_BLOCKED = "roi_gate_blocked"


@dataclass
class BudgeteerEvent:
    """An observable event emitted by the SDK."""

    event_type: str
    run_id: str | None = None
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


EventHook = Callable[[BudgeteerEvent], None]
