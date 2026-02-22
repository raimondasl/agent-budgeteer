"""Reporting module for budget, efficiency, and prediction accuracy metrics.

Produces structured reports from telemetry data, covering:
- Per-run summaries (cost, tokens, latency, success)
- Aggregate statistics by model
- Budget compliance rates
- Prediction accuracy (MAE, bias, MAPE)
- Calibration improvement over time

From the project plan (§9 — Evaluation Plan).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from budgeteer.models import StepMetrics, StepRecord
from budgeteer.telemetry import TelemetryStore


# ------------------------------------------------------------------
# Report data structures
# ------------------------------------------------------------------


@dataclass
class RunSummary:
    """Summary metrics for a single run."""

    run_id: str
    total_cost_usd: float = 0.0
    total_tokens: int = 0
    total_latency_ms: float = 0.0
    total_steps: int = 0
    total_tool_calls: int = 0
    success: bool | None = None


@dataclass
class ModelStats:
    """Aggregate statistics for a single model."""

    model: str
    step_count: int = 0
    total_cost_usd: float = 0.0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_latency_ms: float = 0.0
    avg_cost_per_step: float = 0.0
    avg_latency_per_step: float = 0.0


@dataclass
class PredictionAccuracy:
    """Prediction accuracy metrics for a set of steps.

    Attributes:
        sample_count: Number of steps with both predicted and actual.
        prompt_tokens_mae: Mean absolute error for prompt tokens.
        completion_tokens_mae: Mean absolute error for completion tokens.
        cost_mae: Mean absolute error for cost (USD).
        latency_mae: Mean absolute error for latency (ms).
        cost_bias: Mean signed error for cost (positive = under-predicted).
        latency_bias: Mean signed error for latency.
        cost_mape: Mean absolute percentage error for cost (0–1+).
        latency_mape: Mean absolute percentage error for latency.
    """

    sample_count: int = 0
    prompt_tokens_mae: float = 0.0
    completion_tokens_mae: float = 0.0
    cost_mae: float = 0.0
    latency_mae: float = 0.0
    cost_bias: float = 0.0
    latency_bias: float = 0.0
    cost_mape: float = 0.0
    latency_mape: float = 0.0


@dataclass
class BudgetComplianceReport:
    """Budget compliance summary.

    Attributes:
        total_runs: Total number of completed runs.
        runs_within_budget: Runs that did not exceed any cap.
        runs_exceeded: Runs that exceeded at least one cap.
        compliance_rate: Fraction within budget (0–1).
    """

    total_runs: int = 0
    runs_within_budget: int = 0
    runs_exceeded: int = 0
    compliance_rate: float = 1.0


@dataclass
class FullReport:
    """Complete report combining all sections."""

    run_summaries: list[RunSummary] = field(default_factory=list)
    model_stats: list[ModelStats] = field(default_factory=list)
    prediction_accuracy: PredictionAccuracy = field(default_factory=PredictionAccuracy)
    budget_compliance: BudgetComplianceReport = field(
        default_factory=BudgetComplianceReport
    )


# ------------------------------------------------------------------
# Reporter
# ------------------------------------------------------------------


class Reporter:
    """Generates reports from telemetry data.

    Usage::

        reporter = Reporter(telemetry_store)
        report = reporter.full_report(run_ids=["run-1", "run-2"])
        print(report.prediction_accuracy.cost_mae)
        print(report.budget_compliance.compliance_rate)
    """

    def __init__(self, telemetry: TelemetryStore) -> None:
        self._telemetry = telemetry

    def run_summary(self, run_id: str) -> RunSummary | None:
        """Generate a summary for a single run."""
        run = self._telemetry.get_run(run_id)
        if run is None:
            return None
        return RunSummary(
            run_id=run.run_id,
            total_cost_usd=run.total_cost_usd,
            total_tokens=run.total_tokens,
            total_latency_ms=run.total_latency_ms,
            total_steps=run.total_steps,
            total_tool_calls=run.total_tool_calls,
            success=run.success,
        )

    def model_stats(self, step_records: list[StepRecord]) -> list[ModelStats]:
        """Compute per-model aggregate statistics from step records."""
        by_model: dict[str, ModelStats] = {}

        for rec in step_records:
            model = rec.decision.model
            if model not in by_model:
                by_model[model] = ModelStats(model=model)
            stats = by_model[model]
            stats.step_count += 1

            actual = rec.actual
            if actual is not None:
                stats.total_cost_usd += actual.cost_usd
                stats.total_prompt_tokens += actual.prompt_tokens
                stats.total_completion_tokens += actual.completion_tokens
                stats.total_latency_ms += actual.latency_ms

        for stats in by_model.values():
            if stats.step_count > 0:
                stats.avg_cost_per_step = stats.total_cost_usd / stats.step_count
                stats.avg_latency_per_step = stats.total_latency_ms / stats.step_count

        return sorted(by_model.values(), key=lambda s: s.total_cost_usd, reverse=True)

    def prediction_accuracy(
        self, step_records: list[StepRecord]
    ) -> PredictionAccuracy:
        """Compute prediction accuracy metrics from steps with both predicted and actual."""
        pairs: list[tuple[StepMetrics, StepMetrics]] = []
        for rec in step_records:
            if rec.predicted is not None and rec.actual is not None:
                pairs.append((rec.predicted, rec.actual))

        if not pairs:
            return PredictionAccuracy()

        n = len(pairs)
        pt_ae = ct_ae = cost_ae = lat_ae = 0.0
        cost_se = lat_se = 0.0  # signed errors (for bias)
        cost_ape = lat_ape = 0.0
        cost_ape_count = lat_ape_count = 0

        for pred, actual in pairs:
            pt_ae += abs(actual.prompt_tokens - pred.prompt_tokens)
            ct_ae += abs(actual.completion_tokens - pred.completion_tokens)
            cost_ae += abs(actual.cost_usd - pred.cost_usd)
            lat_ae += abs(actual.latency_ms - pred.latency_ms)

            cost_se += actual.cost_usd - pred.cost_usd
            lat_se += actual.latency_ms - pred.latency_ms

            if pred.cost_usd > 0:
                cost_ape += abs(actual.cost_usd - pred.cost_usd) / pred.cost_usd
                cost_ape_count += 1
            if pred.latency_ms > 0:
                lat_ape += abs(actual.latency_ms - pred.latency_ms) / pred.latency_ms
                lat_ape_count += 1

        return PredictionAccuracy(
            sample_count=n,
            prompt_tokens_mae=pt_ae / n,
            completion_tokens_mae=ct_ae / n,
            cost_mae=cost_ae / n,
            latency_mae=lat_ae / n,
            cost_bias=cost_se / n,
            latency_bias=lat_se / n,
            cost_mape=cost_ape / cost_ape_count if cost_ape_count > 0 else 0.0,
            latency_mape=lat_ape / lat_ape_count if lat_ape_count > 0 else 0.0,
        )

    def budget_compliance(
        self, run_ids: list[str], budget_caps: dict[str, float] | None = None
    ) -> BudgetComplianceReport:
        """Compute budget compliance across runs.

        A run is "within budget" if it completed successfully (success is
        not False) and its total cost doesn't exceed the optional
        ``budget_caps["usd"]`` if provided.

        Args:
            run_ids: Run IDs to check.
            budget_caps: Optional dict with ``"usd"`` and/or ``"tokens"``
                keys specifying the caps to check against.
        """
        total = 0
        within = 0
        exceeded = 0

        for run_id in run_ids:
            run = self._telemetry.get_run(run_id)
            if run is None:
                continue
            total += 1

            violated = False
            if run.success is False:
                # Runs that failed due to budget are counted as exceeded
                violated = True

            if budget_caps:
                if "usd" in budget_caps and run.total_cost_usd > budget_caps["usd"]:
                    violated = True
                if "tokens" in budget_caps and run.total_tokens > budget_caps["tokens"]:
                    violated = True

            if violated:
                exceeded += 1
            else:
                within += 1

        return BudgetComplianceReport(
            total_runs=total,
            runs_within_budget=within,
            runs_exceeded=exceeded,
            compliance_rate=within / total if total > 0 else 1.0,
        )

    def full_report(self, run_ids: list[str], budget_caps: dict[str, float] | None = None) -> FullReport:
        """Generate a comprehensive report for the given runs.

        Combines run summaries, model stats, prediction accuracy, and
        budget compliance into a single :class:`FullReport`.
        """
        summaries: list[RunSummary] = []
        all_steps: list[StepRecord] = []

        for run_id in run_ids:
            s = self.run_summary(run_id)
            if s is not None:
                summaries.append(s)
            steps = self._telemetry.get_steps(run_id)
            all_steps.extend(steps)

        return FullReport(
            run_summaries=summaries,
            model_stats=self.model_stats(all_steps),
            prediction_accuracy=self.prediction_accuracy(all_steps),
            budget_compliance=self.budget_compliance(run_ids, budget_caps),
        )
