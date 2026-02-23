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
    retention_days: int | None = None
    calibration_enabled: bool = True
    calibration_alpha: float = 0.3
    calibration_state_path: str | None = None
    roi_enabled: bool = False
    roi_lambda_latency: float = 0.001
    roi_recommend_threshold: float = 1.0
    roi_budget_floor: float = 0.1
    roi_clarify_ambiguity_threshold: float = 0.6
    degrade_thresholds: tuple[float, float] = (0.8, 0.9)
    degrade_ladder: list[dict] | None = None
    max_retries: int = 0
    retry_delay_ms: float = 1000.0
    fallback_enabled: bool = True

    def validate(self) -> None:
        """Validate all configuration fields, raising ValueError on invalid values."""
        errors: list[str] = []

        if self.default_max_tokens <= 0:
            errors.append("default_max_tokens must be > 0")
        if not (0 < self.calibration_alpha <= 1):
            errors.append("calibration_alpha must be in (0, 1]")
        if self.default_temperature < 0:
            errors.append("default_temperature must be >= 0")
        if self.max_retries < 0:
            errors.append("max_retries must be >= 0")
        if self.retry_delay_ms < 0:
            errors.append("retry_delay_ms must be >= 0")
        if self.roi_lambda_latency < 0:
            errors.append("roi_lambda_latency must be >= 0")
        if self.roi_recommend_threshold <= 0:
            errors.append("roi_recommend_threshold must be > 0")
        if not (0 < self.roi_budget_floor <= 1):
            errors.append("roi_budget_floor must be in (0, 1]")
        if not (0 < self.roi_clarify_ambiguity_threshold <= 1):
            errors.append("roi_clarify_ambiguity_threshold must be in (0, 1]")

        # Validate model tiers
        seen_names: set[str] = set()
        for i, tier in enumerate(self.model_tiers):
            prefix = f"model_tiers[{i}] ({tier.name!r})"
            if not tier.name:
                errors.append(f"model_tiers[{i}]: name must be non-empty")
            if tier.name in seen_names:
                errors.append(f"{prefix}: duplicate model tier name")
            seen_names.add(tier.name)
            if tier.cost_per_prompt_token < 0:
                errors.append(f"{prefix}: cost_per_prompt_token must be >= 0")
            if tier.cost_per_completion_token < 0:
                errors.append(f"{prefix}: cost_per_completion_token must be >= 0")
            if tier.max_context_window <= 0:
                errors.append(f"{prefix}: max_context_window must be > 0")

        if errors:
            raise ValueError("Invalid BudgeteerConfig:\n  - " + "\n  - ".join(errors))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BudgeteerConfig:
        """Create config from a dictionary."""
        tiers_data = data.pop("model_tiers", [])
        budget_data = data.pop("default_run_budget", None)

        tiers = [ModelTier(**t) for t in tiers_data]
        budget = RunBudget(**budget_data) if budget_data else None

        config = cls(model_tiers=tiers, default_run_budget=budget, **data)
        config.validate()
        return config

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

        return cls.from_dict(data)  # from_dict calls validate()
