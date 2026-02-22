"""Configuration loading for Budgeteer.

Supports JSON files natively and YAML files when pyyaml is installed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from budgeteer.models import ModelTier, RunBudget


@dataclass
class BudgeteerConfig:
    """Top-level configuration for the Budgeteer SDK."""

    default_model: str = "gpt-4o-mini"
    default_max_tokens: int = 1024
    default_temperature: float = 0.7
    storage_path: str = "budgeteer_telemetry.db"
    model_tiers: list[ModelTier] = field(default_factory=list)
    default_run_budget: RunBudget | None = None
    calibration_enabled: bool = True
    calibration_alpha: float = 0.3
    roi_enabled: bool = False
    roi_lambda_latency: float = 0.001
    roi_recommend_threshold: float = 1.0
    roi_budget_floor: float = 0.1
    roi_clarify_ambiguity_threshold: float = 0.6
    max_retries: int = 0
    retry_delay_ms: float = 1000.0
    fallback_enabled: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BudgeteerConfig:
        """Create config from a dictionary."""
        tiers_data = data.pop("model_tiers", [])
        budget_data = data.pop("default_run_budget", None)

        tiers = [ModelTier(**t) for t in tiers_data]
        budget = RunBudget(**budget_data) if budget_data else None

        return cls(model_tiers=tiers, default_run_budget=budget, **data)

    @classmethod
    def from_file(cls, path: str | Path) -> BudgeteerConfig:
        """Load config from a JSON or YAML file (auto-detected by extension)."""
        path = Path(path)
        text = path.read_text(encoding="utf-8")

        if path.suffix in (".yaml", ".yml"):
            try:
                import yaml
            except ImportError as exc:
                raise ImportError(
                    "pyyaml is required for YAML config files. "
                    "Install it with: pip install agent-budgeteer[yaml]"
                ) from exc
            data = yaml.safe_load(text)
        else:
            data = json.loads(text)

        return cls.from_dict(data)
