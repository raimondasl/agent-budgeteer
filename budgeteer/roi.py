"""ROI evaluator and clarifying-question selector.

Implements Value-of-Information (VOI) heuristics from the project plan (§6C–D)
to make go/no-go decisions on tool calls, retrieval, and clarifying questions.

The core formula is::

    ROI = benefit_score / (predicted_cost_usd + λ * predicted_latency_ms)

Where benefit is derived from context signals (ambiguity, freshness need,
task complexity, prior failures) and cost comes from the forecaster.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ------------------------------------------------------------------
# Input signals
# ------------------------------------------------------------------


@dataclass
class ROISignals:
    """Context signals used by the ROI evaluator.

    All floats are in the range [0, 1] unless noted otherwise.

    Attributes:
        ambiguity: How unclear the user's intent is (0 = crystal clear,
            1 = very ambiguous).
        freshness_need: How much the task depends on up-to-date data
            (0 = no need, 1 = critical).
        previous_failures: Number of failed tool calls in this run.
        remaining_budget_fraction: Fraction of budget remaining
            (0 = exhausted, 1 = full).
        task_complexity: Estimated complexity of the current step
            (0 = trivial, 1 = very complex).
    """

    ambiguity: float = 0.5
    freshness_need: float = 0.0
    previous_failures: int = 0
    remaining_budget_fraction: float = 1.0
    task_complexity: float = 0.5


# ------------------------------------------------------------------
# Output
# ------------------------------------------------------------------


@dataclass
class ROIResult:
    """Result of an ROI evaluation.

    Attributes:
        roi_score: Computed ROI value (higher = better return).
        recommended: Whether the action is recommended given the score
            and remaining budget.
        reason: Human-readable explanation of the decision.
    """

    roi_score: float
    recommended: bool
    reason: str


# ------------------------------------------------------------------
# Clarifying question templates
# ------------------------------------------------------------------


@dataclass(frozen=True)
class ClarifyingQuestion:
    """A template clarifying question with selection metadata.

    Attributes:
        template: The question text to present to the user.
        category: Semantic category (format, priority, data_source, scope).
        signals: Which signal names make this question most relevant.
            Used for scoring during selection.
    """

    template: str
    category: str
    signals: tuple[str, ...] = ()


DEFAULT_QUESTION_TEMPLATES: list[ClarifyingQuestion] = [
    ClarifyingQuestion(
        template="What output format do you need (JSON / table / text)?",
        category="format",
        signals=("ambiguity",),
    ),
    ClarifyingQuestion(
        template="Is accuracy or speed more important for this task?",
        category="priority",
        signals=("ambiguity", "task_complexity"),
    ),
    ClarifyingQuestion(
        template=(
            "Should I use web search / retrieval, or work only with "
            "the data already provided?"
        ),
        category="data_source",
        signals=("ambiguity", "freshness_need"),
    ),
    ClarifyingQuestion(
        template=(
            "Can you narrow the scope? A more specific request will "
            "produce a faster and cheaper result."
        ),
        category="scope",
        signals=("ambiguity", "task_complexity"),
    ),
]


# ------------------------------------------------------------------
# Evaluator
# ------------------------------------------------------------------


class ROIEvaluator:
    """Value-of-Information evaluator for budget-aware agent decisions.

    Usage::

        evaluator = ROIEvaluator()
        signals = ROISignals(ambiguity=0.8, freshness_need=0.2)

        tool_roi = evaluator.evaluate_tool_call(
            predicted_cost_usd=0.01, predicted_latency_ms=500, signals=signals,
        )
        if not tool_roi.recommended:
            # skip the tool call

        retrieval_roi = evaluator.evaluate_retrieval(
            predicted_cost_usd=0.005, predicted_latency_ms=200, signals=signals,
        )

        question = evaluator.select_question(signals)
        if question is not None:
            # present question.template to the user

    Args:
        lambda_latency: Weight for latency in the cost denominator.
            Higher values penalise slow actions more.
        recommend_threshold: Minimum ROI score to recommend an action.
        budget_floor: Minimum remaining budget fraction below which
            actions are not recommended regardless of ROI.
        clarify_ambiguity_threshold: Minimum ambiguity level to consider
            asking a clarifying question.
    """

    def __init__(
        self,
        lambda_latency: float = 0.001,
        recommend_threshold: float = 1.0,
        budget_floor: float = 0.1,
        clarify_ambiguity_threshold: float = 0.6,
    ) -> None:
        self._lambda = lambda_latency
        self._threshold = recommend_threshold
        self._budget_floor = budget_floor
        self._clarify_threshold = clarify_ambiguity_threshold

    # ------------------------------------------------------------------
    # Tool call ROI
    # ------------------------------------------------------------------

    def evaluate_tool_call(
        self,
        predicted_cost_usd: float,
        predicted_latency_ms: float,
        signals: ROISignals,
    ) -> ROIResult:
        """Evaluate whether a tool call is worth its cost.

        Benefit is driven by task complexity and prior failures (retries
        are more valuable because the agent already committed resources).
        """
        benefit = (
            0.5
            + 0.3 * signals.task_complexity
            + 0.2 * min(1.0, signals.previous_failures / 3)
        )
        return self._evaluate(benefit, predicted_cost_usd, predicted_latency_ms, signals)

    # ------------------------------------------------------------------
    # Retrieval ROI
    # ------------------------------------------------------------------

    def evaluate_retrieval(
        self,
        predicted_cost_usd: float,
        predicted_latency_ms: float,
        signals: ROISignals,
    ) -> ROIResult:
        """Evaluate whether retrieval is worth its cost.

        Benefit is driven by ambiguity (need for more info) and freshness
        (need for up-to-date data).  When both are low, retrieval has
        little expected value.
        """
        benefit = (
            0.3 * signals.ambiguity
            + 0.5 * signals.freshness_need
            + 0.2 * signals.task_complexity
        )
        return self._evaluate(benefit, predicted_cost_usd, predicted_latency_ms, signals)

    # ------------------------------------------------------------------
    # Clarifying question evaluation & selection
    # ------------------------------------------------------------------

    def should_ask_clarification(self, signals: ROISignals) -> ROIResult:
        """Evaluate whether asking a clarifying question is worthwhile.

        Clarifying questions are cheap (no LLM cost) but have latency
        cost (waiting for user response).  They are most valuable when
        ambiguity is high and the task is complex.
        """
        benefit = 0.7 * signals.ambiguity + 0.3 * signals.task_complexity

        # Clarifying questions have zero USD cost and no API latency.
        # Use a small fixed "interruption overhead" cost since the real
        # cost is the user interaction delay, not system resources.
        cost = 0.5
        roi = benefit / cost

        recommended = (
            roi >= self._threshold
            and signals.ambiguity >= self._clarify_threshold
            and signals.remaining_budget_fraction >= self._budget_floor
        )

        if recommended:
            reason = (
                f"Clarification recommended: ambiguity={signals.ambiguity:.1f}, "
                f"ROI={roi:.2f}"
            )
        elif signals.ambiguity < self._clarify_threshold:
            reason = f"Ambiguity too low ({signals.ambiguity:.1f}) to justify clarification"
        else:
            reason = f"Clarification ROI ({roi:.2f}) below threshold"

        return ROIResult(roi_score=roi, recommended=recommended, reason=reason)

    def select_question(
        self,
        signals: ROISignals,
        templates: list[ClarifyingQuestion] | None = None,
        exclude_categories: set[str] | None = None,
    ) -> ClarifyingQuestion | None:
        """Select the most relevant clarifying question, or ``None``.

        Returns ``None`` if clarification is not recommended (based on
        :meth:`should_ask_clarification`) or if no templates match.

        Args:
            signals: Current context signals.
            templates: Custom template list.  Defaults to
                :data:`DEFAULT_QUESTION_TEMPLATES`.
            exclude_categories: Categories to skip (e.g. already asked).
        """
        check = self.should_ask_clarification(signals)
        if not check.recommended:
            return None

        pool = templates if templates is not None else DEFAULT_QUESTION_TEMPLATES
        exclude = exclude_categories or set()

        scored: list[tuple[float, ClarifyingQuestion]] = []
        for q in pool:
            if q.category in exclude:
                continue
            score = self._score_question(q, signals)
            scored.append((score, q))

        if not scored:
            return None

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _evaluate(
        self,
        benefit: float,
        predicted_cost_usd: float,
        predicted_latency_ms: float,
        signals: ROISignals,
    ) -> ROIResult:
        """Core ROI calculation shared by tool-call and retrieval evaluators."""
        cost = predicted_cost_usd + self._lambda * predicted_latency_ms

        # Avoid division by zero for free actions
        roi = benefit / cost if cost > 0 else benefit * 100

        recommended = (
            roi >= self._threshold
            and signals.remaining_budget_fraction >= self._budget_floor
        )

        if recommended:
            reason = f"ROI={roi:.2f} (benefit={benefit:.2f}, cost=${cost:.4f})"
        elif signals.remaining_budget_fraction < self._budget_floor:
            reason = (
                f"Budget too low ({signals.remaining_budget_fraction:.0%}) "
                f"— skipping despite ROI={roi:.2f}"
            )
        else:
            reason = f"ROI={roi:.2f} below threshold {self._threshold}"

        return ROIResult(roi_score=roi, recommended=recommended, reason=reason)

    @staticmethod
    def _score_question(q: ClarifyingQuestion, signals: ROISignals) -> float:
        """Score a question template against the current signals.

        Sums the matching signal values — questions that reference more
        relevant signals naturally score higher.
        """
        signal_map = {
            "ambiguity": signals.ambiguity,
            "freshness_need": signals.freshness_need,
            "task_complexity": signals.task_complexity,
        }
        return sum(signal_map.get(s, 0.0) for s in q.signals)
