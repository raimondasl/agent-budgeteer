"""Calibration engine for improving prediction accuracy over time.

Compares predicted vs actual metrics from step records and maintains
per-model exponential moving average (EMA) correction factors.  These
factors can be applied to future forecasts so that predictions converge
toward reality as more data is collected.

From the project plan (§6A):
  - Per-model correction factors (moving average error)
  - Per-task-type completion distributions
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from budgeteer.models import StepMetrics, StepRecord


# ------------------------------------------------------------------
# Correction factors
# ------------------------------------------------------------------


@dataclass
class CorrectionFactors:
    """Per-model multiplicative correction factors.

    A factor of 1.0 means no correction.  A factor of 1.2 means the
    predictor under-estimates by ~20%, so forecasts should be multiplied
    by 1.2.

    Attributes:
        prompt_tokens: Correction for prompt token predictions.
        completion_tokens: Correction for completion token predictions.
        cost_usd: Correction for cost predictions.
        latency_ms: Correction for latency predictions.
        sample_count: Number of observations used to compute these factors.
    """

    prompt_tokens: float = 1.0
    completion_tokens: float = 1.0
    cost_usd: float = 1.0
    latency_ms: float = 1.0
    sample_count: int = 0


# ------------------------------------------------------------------
# Prediction error record
# ------------------------------------------------------------------


@dataclass
class PredictionError:
    """Error between predicted and actual for a single step.

    Ratios > 1.0 mean the predictor under-estimated (actual > predicted).
    Ratios < 1.0 mean the predictor over-estimated.
    """

    prompt_tokens_ratio: float | None = None
    completion_tokens_ratio: float | None = None
    cost_ratio: float | None = None
    latency_ratio: float | None = None


# ------------------------------------------------------------------
# Calibrator
# ------------------------------------------------------------------


class Calibrator:
    """Maintains and applies per-model correction factors.

    Usage::

        cal = Calibrator(alpha=0.3)

        # After each step with predictions:
        cal.update("gpt-4o", predicted=step.predicted, actual=step.actual)

        # Apply corrections to future forecasts:
        factors = cal.get_factors("gpt-4o")
        corrected_cost = raw_forecast * factors.cost_usd

    Args:
        alpha: EMA smoothing factor (0 < alpha <= 1).
            Higher values weight recent observations more.
            Default 0.3 gives a good balance of responsiveness
            and stability.
    """

    def __init__(self, alpha: float = 0.3) -> None:
        if not 0 < alpha <= 1:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self._alpha = alpha
        self._factors: dict[str, CorrectionFactors] = {}

    @property
    def models(self) -> list[str]:
        """List of models with calibration data."""
        return list(self._factors.keys())

    def get_factors(self, model: str) -> CorrectionFactors:
        """Return correction factors for *model*, or defaults if unseen."""
        return self._factors.get(model, CorrectionFactors())

    def update(
        self, model: str, predicted: StepMetrics, actual: StepMetrics
    ) -> PredictionError:
        """Update correction factors with a new predicted/actual observation.

        Returns the :class:`PredictionError` for this observation.
        """
        error = self._compute_error(predicted, actual)
        factors = self._factors.get(model, CorrectionFactors())

        factors.prompt_tokens = self._ema_update(
            factors.prompt_tokens, error.prompt_tokens_ratio
        )
        factors.completion_tokens = self._ema_update(
            factors.completion_tokens, error.completion_tokens_ratio
        )
        factors.cost_usd = self._ema_update(
            factors.cost_usd, error.cost_ratio
        )
        factors.latency_ms = self._ema_update(
            factors.latency_ms, error.latency_ratio
        )
        factors.sample_count += 1

        self._factors[model] = factors
        return error

    def update_from_record(self, record: StepRecord) -> PredictionError | None:
        """Convenience: update from a step record if it has both predicted and actual."""
        if record.predicted is None or record.actual is None:
            return None
        return self.update(
            record.decision.model, record.predicted, record.actual
        )

    def apply(self, model: str, predicted: StepMetrics) -> StepMetrics:
        """Return a corrected copy of *predicted* using the model's factors.

        If no calibration data exists for the model, returns an
        unmodified copy.
        """
        factors = self.get_factors(model)
        return StepMetrics(
            prompt_tokens=max(1, round(predicted.prompt_tokens * factors.prompt_tokens)),
            completion_tokens=max(1, round(predicted.completion_tokens * factors.completion_tokens)),
            cost_usd=predicted.cost_usd * factors.cost_usd,
            latency_ms=predicted.latency_ms * factors.latency_ms,
            tool_calls_made=predicted.tool_calls_made,
            success=predicted.success,
        )

    def bulk_update(self, records: list[StepRecord]) -> int:
        """Update factors from a batch of step records.

        Returns the number of records that had both predicted and actual.
        """
        count = 0
        for rec in records:
            if self.update_from_record(rec) is not None:
                count += 1
        return count

    def save(self, path: str | Path) -> None:
        """Serialize correction factors to a JSON file."""
        data: dict[str, dict[str, float | int]] = {}
        for model, factors in self._factors.items():
            data[model] = {
                "prompt_tokens": factors.prompt_tokens,
                "completion_tokens": factors.completion_tokens,
                "cost_usd": factors.cost_usd,
                "latency_ms": factors.latency_ms,
                "sample_count": factors.sample_count,
            }
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    def load(self, path: str | Path) -> None:
        """Load correction factors from a JSON file, merging with existing.

        If the file does not exist, does nothing (first-run scenario).
        Raises ValueError on corrupt/unparseable JSON.
        """
        p = Path(path)
        if not p.exists():
            return
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError(f"Corrupt calibration state file: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError("Calibration state file must contain a JSON object")
        for model, factors_dict in data.items():
            self._factors[model] = CorrectionFactors(
                prompt_tokens=float(factors_dict.get("prompt_tokens", 1.0)),
                completion_tokens=float(factors_dict.get("completion_tokens", 1.0)),
                cost_usd=float(factors_dict.get("cost_usd", 1.0)),
                latency_ms=float(factors_dict.get("latency_ms", 1.0)),
                sample_count=int(factors_dict.get("sample_count", 0)),
            )

    @classmethod
    def from_file(cls, path: str | Path, alpha: float = 0.3) -> Calibrator:
        """Construct a Calibrator with pre-loaded factors from a file.

        If the file does not exist, returns a fresh Calibrator.
        """
        cal = cls(alpha=alpha)
        cal.load(path)
        return cal

    def reset(self, model: str | None = None) -> None:
        """Reset correction factors.

        If *model* is given, reset only that model.
        Otherwise reset all models.
        """
        if model is not None:
            self._factors.pop(model, None)
        else:
            self._factors.clear()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_error(
        predicted: StepMetrics, actual: StepMetrics
    ) -> PredictionError:
        """Compute per-metric ratios (actual / predicted)."""

        def _ratio(pred: float | int, act: float | int) -> float | None:
            if pred == 0:
                return None  # cannot compute ratio
            return act / pred

        return PredictionError(
            prompt_tokens_ratio=_ratio(predicted.prompt_tokens, actual.prompt_tokens),
            completion_tokens_ratio=_ratio(
                predicted.completion_tokens, actual.completion_tokens
            ),
            cost_ratio=_ratio(predicted.cost_usd, actual.cost_usd),
            latency_ratio=_ratio(predicted.latency_ms, actual.latency_ms),
        )

    def _ema_update(
        self, current: float, new_ratio: float | None
    ) -> float:
        """Exponential moving average update.

        If *new_ratio* is None (e.g. division by zero), keep current.
        """
        if new_ratio is None:
            return current
        return current * (1 - self._alpha) + new_ratio * self._alpha
