"""Tests for budgeteer.roi — ROI evaluation and clarifying question selection."""

import pytest

from budgeteer.roi import (
    DEFAULT_QUESTION_TEMPLATES,
    ClarifyingQuestion,
    ROIEvaluator,
    ROIResult,
    ROISignals,
)


# ===========================================================================
# ROISignals defaults
# ===========================================================================


class TestROISignals:
    def test_defaults(self):
        s = ROISignals()
        assert s.ambiguity == 0.5
        assert s.freshness_need == 0.0
        assert s.previous_failures == 0
        assert s.remaining_budget_fraction == 1.0
        assert s.task_complexity == 0.5

    def test_custom(self):
        s = ROISignals(
            ambiguity=0.9,
            freshness_need=0.8,
            previous_failures=2,
            remaining_budget_fraction=0.3,
            task_complexity=0.7,
        )
        assert s.ambiguity == 0.9
        assert s.previous_failures == 2


# ===========================================================================
# ROIResult
# ===========================================================================


class TestROIResult:
    def test_creation(self):
        r = ROIResult(roi_score=5.0, recommended=True, reason="good")
        assert r.roi_score == 5.0
        assert r.recommended is True
        assert r.reason == "good"


# ===========================================================================
# Tool call ROI evaluation
# ===========================================================================


class TestToolCallROI:
    def test_cheap_tool_high_complexity_recommended(self):
        ev = ROIEvaluator()
        signals = ROISignals(task_complexity=0.8, remaining_budget_fraction=0.9)
        result = ev.evaluate_tool_call(
            predicted_cost_usd=0.001, predicted_latency_ms=100, signals=signals,
        )
        assert result.recommended is True
        assert result.roi_score > 1.0

    def test_expensive_tool_low_complexity_not_recommended(self):
        ev = ROIEvaluator()
        signals = ROISignals(task_complexity=0.1, remaining_budget_fraction=0.5)
        result = ev.evaluate_tool_call(
            predicted_cost_usd=1.0, predicted_latency_ms=5000, signals=signals,
        )
        assert result.recommended is False

    def test_previous_failures_increase_benefit(self):
        ev = ROIEvaluator()
        no_fail = ROISignals(previous_failures=0)
        with_fail = ROISignals(previous_failures=3)

        r1 = ev.evaluate_tool_call(0.01, 200, no_fail)
        r2 = ev.evaluate_tool_call(0.01, 200, with_fail)
        assert r2.roi_score > r1.roi_score

    def test_low_budget_blocks_even_high_roi(self):
        ev = ROIEvaluator()
        signals = ROISignals(
            task_complexity=1.0,
            remaining_budget_fraction=0.05,  # below floor
        )
        result = ev.evaluate_tool_call(0.0001, 10, signals)
        assert result.recommended is False
        assert "Budget too low" in result.reason

    def test_zero_cost_tool(self):
        ev = ROIEvaluator()
        signals = ROISignals()
        result = ev.evaluate_tool_call(0.0, 0.0, signals)
        # Free tool → very high ROI
        assert result.roi_score > 10
        assert result.recommended is True

    def test_reason_contains_roi_info(self):
        ev = ROIEvaluator()
        signals = ROISignals()
        result = ev.evaluate_tool_call(0.01, 200, signals)
        assert "ROI=" in result.reason


# ===========================================================================
# Retrieval ROI evaluation
# ===========================================================================


class TestRetrievalROI:
    def test_high_freshness_recommends_retrieval(self):
        ev = ROIEvaluator()
        signals = ROISignals(freshness_need=0.9, remaining_budget_fraction=0.8)
        result = ev.evaluate_retrieval(0.005, 200, signals)
        assert result.recommended is True

    def test_low_freshness_low_ambiguity_skips_retrieval(self):
        """This is the 'skip retrieval if low ROI' guard from the plan."""
        ev = ROIEvaluator()
        signals = ROISignals(
            ambiguity=0.1,
            freshness_need=0.0,
            task_complexity=0.2,
        )
        result = ev.evaluate_retrieval(0.01, 500, signals)
        assert result.recommended is False
        assert "below threshold" in result.reason

    def test_high_ambiguity_increases_retrieval_roi(self):
        ev = ROIEvaluator()
        low_amb = ROISignals(ambiguity=0.1, freshness_need=0.3)
        high_amb = ROISignals(ambiguity=0.9, freshness_need=0.3)

        r1 = ev.evaluate_retrieval(0.01, 200, low_amb)
        r2 = ev.evaluate_retrieval(0.01, 200, high_amb)
        assert r2.roi_score > r1.roi_score

    def test_retrieval_benefit_formula(self):
        """Verify: benefit = 0.3*ambiguity + 0.5*freshness + 0.2*complexity."""
        ev = ROIEvaluator(lambda_latency=0)
        signals = ROISignals(ambiguity=1.0, freshness_need=1.0, task_complexity=1.0)
        result = ev.evaluate_retrieval(1.0, 0, signals)
        # benefit = 0.3 + 0.5 + 0.2 = 1.0; cost = 1.0; ROI = 1.0
        assert result.roi_score == pytest.approx(1.0)

    def test_expensive_retrieval_low_budget_skipped(self):
        ev = ROIEvaluator()
        signals = ROISignals(
            freshness_need=0.9,
            remaining_budget_fraction=0.05,
        )
        result = ev.evaluate_retrieval(0.5, 1000, signals)
        assert result.recommended is False


# ===========================================================================
# Clarifying question evaluation
# ===========================================================================


class TestClarifyingQuestion:
    def test_high_ambiguity_recommends(self):
        ev = ROIEvaluator()
        signals = ROISignals(ambiguity=0.9, task_complexity=0.8)
        result = ev.should_ask_clarification(signals)
        assert result.recommended is True

    def test_low_ambiguity_does_not_recommend(self):
        ev = ROIEvaluator()
        signals = ROISignals(ambiguity=0.2)
        result = ev.should_ask_clarification(signals)
        assert result.recommended is False
        assert "Ambiguity too low" in result.reason

    def test_ambiguity_threshold_configurable(self):
        ev = ROIEvaluator(clarify_ambiguity_threshold=0.3)
        signals = ROISignals(ambiguity=0.4, task_complexity=0.8)
        result = ev.should_ask_clarification(signals)
        assert result.recommended is True

    def test_low_budget_blocks_clarification(self):
        ev = ROIEvaluator()
        signals = ROISignals(
            ambiguity=0.9,
            remaining_budget_fraction=0.05,
        )
        result = ev.should_ask_clarification(signals)
        assert result.recommended is False


# ===========================================================================
# Question selection
# ===========================================================================


class TestQuestionSelection:
    def test_selects_question_when_ambiguity_high(self):
        ev = ROIEvaluator()
        signals = ROISignals(ambiguity=0.9, task_complexity=0.7)
        q = ev.select_question(signals)
        assert q is not None
        assert isinstance(q, ClarifyingQuestion)
        assert len(q.template) > 0

    def test_returns_none_when_ambiguity_low(self):
        ev = ROIEvaluator()
        signals = ROISignals(ambiguity=0.2)
        assert ev.select_question(signals) is None

    def test_freshness_signal_prefers_data_source_question(self):
        ev = ROIEvaluator()
        signals = ROISignals(
            ambiguity=0.9,
            freshness_need=0.9,
            task_complexity=0.1,
        )
        q = ev.select_question(signals)
        assert q is not None
        assert q.category == "data_source"

    def test_high_complexity_prefers_priority_or_scope(self):
        ev = ROIEvaluator()
        signals = ROISignals(
            ambiguity=0.9,
            freshness_need=0.0,
            task_complexity=0.9,
        )
        q = ev.select_question(signals)
        assert q is not None
        assert q.category in ("priority", "scope")

    def test_exclude_categories(self):
        ev = ROIEvaluator()
        signals = ROISignals(ambiguity=0.9, task_complexity=0.9)
        q = ev.select_question(signals, exclude_categories={"priority", "scope"})
        assert q is not None
        assert q.category not in ("priority", "scope")

    def test_custom_templates(self):
        custom = [
            ClarifyingQuestion(
                template="Do you want verbose output?",
                category="verbosity",
                signals=("ambiguity",),
            ),
        ]
        ev = ROIEvaluator()
        signals = ROISignals(ambiguity=0.9)
        q = ev.select_question(signals, templates=custom)
        assert q is not None
        assert q.category == "verbosity"

    def test_all_categories_excluded_returns_none(self):
        ev = ROIEvaluator()
        signals = ROISignals(ambiguity=0.9)
        all_cats = {q.category for q in DEFAULT_QUESTION_TEMPLATES}
        assert ev.select_question(signals, exclude_categories=all_cats) is None

    def test_returns_none_when_budget_low(self):
        ev = ROIEvaluator()
        signals = ROISignals(ambiguity=0.9, remaining_budget_fraction=0.01)
        assert ev.select_question(signals) is None


# ===========================================================================
# Default question templates
# ===========================================================================


class TestDefaultTemplates:
    def test_four_templates(self):
        assert len(DEFAULT_QUESTION_TEMPLATES) == 4

    def test_unique_categories(self):
        cats = [q.category for q in DEFAULT_QUESTION_TEMPLATES]
        assert len(cats) == len(set(cats))

    def test_all_have_signals(self):
        for q in DEFAULT_QUESTION_TEMPLATES:
            assert len(q.signals) > 0

    def test_templates_are_frozen(self):
        q = DEFAULT_QUESTION_TEMPLATES[0]
        with pytest.raises(AttributeError):
            q.template = "changed"


# ===========================================================================
# Evaluator configuration
# ===========================================================================


class TestEvaluatorConfig:
    def test_custom_lambda_affects_latency_sensitivity(self):
        low_lambda = ROIEvaluator(lambda_latency=0.0001)
        high_lambda = ROIEvaluator(lambda_latency=0.01)
        signals = ROISignals()

        r1 = low_lambda.evaluate_tool_call(0.01, 1000, signals)
        r2 = high_lambda.evaluate_tool_call(0.01, 1000, signals)
        # Higher lambda → higher effective cost → lower ROI
        assert r1.roi_score > r2.roi_score

    def test_custom_threshold(self):
        strict = ROIEvaluator(recommend_threshold=5.0)
        lenient = ROIEvaluator(recommend_threshold=0.1)
        signals = ROISignals()

        r1 = strict.evaluate_tool_call(0.05, 500, signals)
        r2 = lenient.evaluate_tool_call(0.05, 500, signals)
        # Same ROI, but strict may reject while lenient accepts
        assert r1.roi_score == r2.roi_score
        assert r2.recommended is True
        # Strict may or may not be recommended depending on the actual ROI

    def test_custom_budget_floor(self):
        ev = ROIEvaluator(budget_floor=0.5)
        signals = ROISignals(remaining_budget_fraction=0.3)
        result = ev.evaluate_tool_call(0.001, 100, signals)
        assert result.recommended is False
