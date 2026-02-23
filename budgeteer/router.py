"""Strategy router for candidate generation, cost forecasting, and selection.

Generates candidate strategies from configured model tiers at various
degradation levels, estimates their resource usage, and selects the
optimal strategy that fits within budget constraints.

This implements the control loop from the project plan (§5):
  1. Build candidate strategies
  2. Forecast per candidate
  3. Apply policy constraints (filter infeasible)
  4. Choose best candidate under budgets
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from budgeteer.config import BudgeteerConfig
from budgeteer.models import ModelTier, StepContext, StepDecision, StepMetrics

if TYPE_CHECKING:
    from budgeteer.calibrator import Calibrator


# ------------------------------------------------------------------
# Degradation ladder (project plan §7)
# ------------------------------------------------------------------


@dataclass(frozen=True)
class DegradeLevelParams:
    """Configuration for a single level in the degradation ladder."""

    max_tokens_ratio: float
    tool_calls: int
    retrieval: bool
    retrieval_top_k: int
    quality: float


DEFAULT_DEGRADE_LADDER: list[DegradeLevelParams] = [
    # Level 0 — Full: strong model + retrieval + tools
    DegradeLevelParams(max_tokens_ratio=1.0, tool_calls=5, retrieval=True, retrieval_top_k=3, quality=1.0),
    # Level 1 — Reduced retrieval (top-k down)
    DegradeLevelParams(max_tokens_ratio=0.75, tool_calls=3, retrieval=True, retrieval_top_k=1, quality=0.85),
    # Level 2 — Smaller model preferred, minimal retrieval
    DegradeLevelParams(max_tokens_ratio=0.5, tool_calls=2, retrieval=True, retrieval_top_k=1, quality=0.7),
    # Level 3 — No retrieval, internal reasoning only
    DegradeLevelParams(max_tokens_ratio=0.5, tool_calls=1, retrieval=False, retrieval_top_k=0, quality=0.5),
    # Level 4 — Strict summarization (minimal tokens, no tools)
    DegradeLevelParams(max_tokens_ratio=0.25, tool_calls=0, retrieval=False, retrieval_top_k=0, quality=0.3),
]


# ------------------------------------------------------------------
# Candidate strategy
# ------------------------------------------------------------------


@dataclass
class CandidateStrategy:
    """A candidate execution strategy with predicted resource usage.

    Produced by :meth:`StrategyRouter.generate_candidates` and enriched
    by :meth:`StrategyRouter.forecast`.
    """

    model: str
    max_tokens: int
    temperature: float
    context_window: int
    tool_calls_allowed: int
    retrieval_enabled: bool
    retrieval_top_k: int
    degrade_level: int

    # Forecasted resource usage (filled by forecast())
    predicted_prompt_tokens: int = 0
    predicted_completion_tokens: int = 0
    predicted_cost_usd: float = 0.0
    predicted_latency_ms: float = 0.0

    # Quality estimate (0–1, where 1 = highest quality)
    quality_score: float = 1.0


# ------------------------------------------------------------------
# Strategy router
# ------------------------------------------------------------------


class StrategyRouter:
    """Generates, forecasts, and selects candidate strategies.

    When configured with model tiers, the router produces candidates at
    multiple degradation levels for each tier and can select the best
    feasible strategy under budget constraints.

    Usage::

        router = StrategyRouter(config)
        candidates = router.generate_candidates(context)
        for c in candidates:
            router.forecast(c, context)
        best = router.select(candidates, remaining_usd=0.50)
        decision = router.to_decision(best)
    """

    def __init__(self, config: BudgeteerConfig, calibrator: "Calibrator | None" = None) -> None:
        """Initialize with model tiers sorted by cost."""
        self._config = config
        self._calibrator = calibrator
        self._tiers_by_cost: list[ModelTier] = sorted(
            config.model_tiers,
            key=lambda t: t.cost_per_prompt_token + t.cost_per_completion_token,
        )
        # Use custom degradation ladder if provided
        if config.degrade_ladder is not None:
            self._degrade_ladder = self._parse_ladder(config.degrade_ladder)
        else:
            self._degrade_ladder = DEFAULT_DEGRADE_LADDER

    @staticmethod
    def _parse_ladder(ladder_dicts: list[dict]) -> list[DegradeLevelParams]:
        """Parse and validate a custom degradation ladder from config dicts."""
        if not ladder_dicts:
            raise ValueError("degrade_ladder must have at least 1 level")
        result = []
        for i, d in enumerate(ladder_dicts):
            mtr = d.get("max_tokens_ratio", 1.0)
            quality = d.get("quality", 1.0)
            if not (0 <= mtr <= 1):
                raise ValueError(f"degrade_ladder[{i}]: max_tokens_ratio must be in [0, 1], got {mtr}")
            if not (0 <= quality <= 1):
                raise ValueError(f"degrade_ladder[{i}]: quality must be in [0, 1], got {quality}")
            result.append(DegradeLevelParams(
                max_tokens_ratio=mtr,
                tool_calls=int(d.get("tool_calls", 5)),
                retrieval=bool(d.get("retrieval", True)),
                retrieval_top_k=int(d.get("retrieval_top_k", 3)),
                quality=quality,
            ))
        return result

    def set_calibrator(self, calibrator: "Calibrator | None") -> None:
        """Set or replace the calibrator used for forecast correction."""
        self._calibrator = calibrator

    @property
    def available(self) -> bool:
        """True if at least one model tier is configured for routing."""
        return len(self._tiers_by_cost) >= 1

    # ------------------------------------------------------------------
    # 1. Candidate generation
    # ------------------------------------------------------------------

    def generate_candidates(
        self, context: StepContext
    ) -> list[CandidateStrategy]:
        """Build candidate strategies from model tiers x degradation levels.

        Produces ``len(tiers) * 5`` candidates.  Quality scores reflect
        both the model's capability (more expensive = higher tier quality)
        and the degradation level.
        """
        candidates: list[CandidateStrategy] = []
        n = len(self._tiers_by_cost)

        for tier_idx, tier in enumerate(self._tiers_by_cost):
            # Tier quality: cheapest → 0.6, most expensive → 1.0 (linear)
            tier_quality = (
                (0.6 + 0.4 * tier_idx / max(1, n - 1)) if n > 1 else 1.0
            )

            for level, params in enumerate(self._degrade_ladder):
                max_tokens = max(
                    1,
                    int(self._config.default_max_tokens * params.max_tokens_ratio),
                )
                candidates.append(
                    CandidateStrategy(
                        model=tier.name,
                        max_tokens=max_tokens,
                        temperature=self._config.default_temperature,
                        context_window=tier.max_context_window,
                        tool_calls_allowed=params.tool_calls,
                        retrieval_enabled=params.retrieval,
                        retrieval_top_k=params.retrieval_top_k,
                        degrade_level=level,
                        quality_score=tier_quality * params.quality,
                    )
                )

        return candidates

    # ------------------------------------------------------------------
    # 2. Forecasting
    # ------------------------------------------------------------------

    def forecast(
        self, candidate: CandidateStrategy, context: StepContext
    ) -> None:
        """Estimate resource usage for *candidate*.  Mutates in place.

        Uses the model tier's pricing to calculate cost, and simple
        heuristics for prompt-token count and latency.  Completion tokens
        are conservatively estimated as ``max_tokens`` (worst case).

        When a calibrator is configured, raw predictions are corrected
        using learned per-model factors.
        """
        tier = self._get_tier(candidate.model)
        if tier is None:
            return

        prompt_tokens = self._estimate_prompt_tokens(context)
        completion_tokens = candidate.max_tokens  # conservative worst-case

        raw_cost = (
            prompt_tokens * tier.cost_per_prompt_token
            + completion_tokens * tier.cost_per_completion_token
        )
        raw_latency = 30.0 + completion_tokens * 0.015

        # Apply calibration corrections if available
        if self._calibrator is not None:
            raw_metrics = StepMetrics(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost_usd=raw_cost,
                latency_ms=raw_latency,
            )
            corrected = self._calibrator.apply(candidate.model, raw_metrics)
            candidate.predicted_prompt_tokens = corrected.prompt_tokens
            candidate.predicted_completion_tokens = corrected.completion_tokens
            candidate.predicted_cost_usd = corrected.cost_usd
            candidate.predicted_latency_ms = corrected.latency_ms
        else:
            candidate.predicted_prompt_tokens = prompt_tokens
            candidate.predicted_completion_tokens = completion_tokens
            candidate.predicted_cost_usd = raw_cost
            candidate.predicted_latency_ms = raw_latency

    def get_prediction(self, candidate: CandidateStrategy) -> StepMetrics:
        """Convert a candidate's forecast fields into a StepMetrics."""
        return StepMetrics(
            prompt_tokens=candidate.predicted_prompt_tokens,
            completion_tokens=candidate.predicted_completion_tokens,
            cost_usd=candidate.predicted_cost_usd,
            latency_ms=candidate.predicted_latency_ms,
        )

    # ------------------------------------------------------------------
    # 3. Selection
    # ------------------------------------------------------------------

    def select(
        self,
        candidates: list[CandidateStrategy],
        remaining_usd: float | None = None,
        remaining_tokens: int | None = None,
        remaining_latency_ms: float | None = None,
    ) -> CandidateStrategy | None:
        """Pick the highest-quality candidate within budget constraints.

        Each constraint is only applied when its value is not ``None``.
        Returns ``None`` when no candidate fits within all constraints.
        """
        feasible: list[CandidateStrategy] = []
        for c in candidates:
            if remaining_usd is not None and c.predicted_cost_usd > remaining_usd:
                continue
            if remaining_tokens is not None:
                total = c.predicted_prompt_tokens + c.predicted_completion_tokens
                if total > remaining_tokens:
                    continue
            if (
                remaining_latency_ms is not None
                and c.predicted_latency_ms > remaining_latency_ms
            ):
                continue
            feasible.append(c)

        if not feasible:
            return None

        return max(feasible, key=lambda c: c.quality_score)

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def to_decision(self, candidate: CandidateStrategy) -> StepDecision:
        """Convert a selected :class:`CandidateStrategy` into a :class:`StepDecision`."""
        return StepDecision(
            model=candidate.model,
            max_tokens=candidate.max_tokens,
            temperature=candidate.temperature,
            context_window=candidate.context_window,
            tool_calls_allowed=candidate.tool_calls_allowed,
            retrieval_enabled=candidate.retrieval_enabled,
            retrieval_top_k=candidate.retrieval_top_k,
            degrade_level=candidate.degrade_level,
            degrade_reason=(
                f"routed: degrade level {candidate.degrade_level}"
                if candidate.degrade_level > 0
                else None
            ),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_tier(self, model: str) -> ModelTier | None:
        for t in self._tiers_by_cost:
            if t.name == model:
                return t
        return None

    def _estimate_prompt_tokens(self, context: StepContext) -> int:
        """Rough token estimate: ~4 characters per token for English text."""
        if not context.messages:
            return 100  # baseline for empty/unknown context
        total_chars = sum(
            len(str(m.get("content", ""))) for m in context.messages
        )
        return max(1, total_chars // 4)
