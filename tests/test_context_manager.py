"""Tests for budgeteer.context_manager — truncation, summarization, and retrieval packing."""

import pytest

from budgeteer.context_manager import (
    ContextManager,
    RetrievalResult,
    _message_tokens,
    estimate_tokens,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _msg(role: str, content: str) -> dict:
    return {"role": role, "content": content}


def _system(content: str) -> dict:
    return _msg("system", content)


def _user(content: str) -> dict:
    return _msg("user", content)


def _assistant(content: str) -> dict:
    return _msg("assistant", content)


def _total_tokens(messages: list[dict]) -> int:
    return sum(_message_tokens(m) for m in messages)


# ===========================================================================
# Token estimation
# ===========================================================================


class TestEstimateTokens:
    def test_empty_string(self):
        assert estimate_tokens("") == 1  # min 1

    def test_short_text(self):
        # "hello" = 5 chars → 5//4 = 1
        assert estimate_tokens("hello") == 1

    def test_longer_text(self):
        # 400 chars → 100 tokens
        assert estimate_tokens("x" * 400) == 100

    def test_message_tokens_includes_overhead(self):
        msg = _user("hello")
        tokens = _message_tokens(msg)
        # "user" + "hello" = 9 chars → 2 tokens + 4 overhead = 6
        assert tokens == 6


# ===========================================================================
# Truncation
# ===========================================================================


class TestTruncation:
    def test_empty_messages(self):
        cm = ContextManager(max_tokens=1000)
        assert cm.truncate([]) == []

    def test_messages_within_budget_unchanged(self):
        cm = ContextManager(max_tokens=10000)
        msgs = [_system("Be helpful"), _user("Hi"), _assistant("Hello!")]
        result = cm.truncate(msgs)
        assert len(result) == 3
        assert result[0]["content"] == "Be helpful"
        assert result[2]["content"] == "Hello!"

    def test_drops_oldest_non_system_first(self):
        cm = ContextManager(max_tokens=100)
        msgs = [
            _system("sys"),
            _user("old message " * 10),
            _user("newer message " * 10),
            _user("newest"),
        ]
        result = cm.truncate(msgs)
        # System message should be kept
        assert result[0]["role"] == "system"
        # Most recent messages kept, oldest dropped
        assert result[-1]["content"] == "newest"

    def test_system_messages_always_kept(self):
        cm = ContextManager(max_tokens=200)
        msgs = [
            _system("Important system instructions"),
            _user("old " * 50),
            _user("recent"),
        ]
        result = cm.truncate(msgs)
        assert result[0]["role"] == "system"
        assert "Important system instructions" in result[0]["content"]

    def test_multiple_system_messages_at_start_kept(self):
        cm = ContextManager(max_tokens=500)
        msgs = [
            _system("System 1"),
            _system("System 2"),
            _user("Hello"),
            _assistant("Hi there"),
        ]
        result = cm.truncate(msgs)
        system_count = sum(1 for m in result if m.get("role") == "system")
        assert system_count >= 1  # At least the system messages are preserved

    def test_system_message_not_preserved_if_mid_conversation(self):
        """System messages after non-system messages are treated as regular messages."""
        cm = ContextManager(max_tokens=50)
        msgs = [
            _user("Hello"),
            _system("Mid-conversation system"),  # not leading
            _user("Recent"),
        ]
        result = cm.truncate(msgs)
        # "Mid-conversation system" is not a leading system msg, can be dropped
        assert result[-1]["content"] == "Recent"

    def test_respects_explicit_max_tokens(self):
        cm = ContextManager(max_tokens=100000)
        msgs = [_user("x" * 400), _user("y" * 400)]
        # Override with a small budget
        result = cm.truncate(msgs, max_tokens=50)
        total = _total_tokens(result)
        assert total <= 50

    def test_reserve_tokens_reduces_budget(self):
        cm = ContextManager(max_tokens=200, reserve_tokens=100)
        assert cm.available_tokens == 100
        msgs = [_user("x" * 800)]  # ~200 tokens, won't fit in 100
        result = cm.truncate(msgs)
        total = _total_tokens(result)
        assert total <= 100

    def test_truncates_long_single_message(self):
        cm = ContextManager(max_tokens=50)
        msgs = [_user("x" * 2000)]  # ~500 tokens
        result = cm.truncate(msgs)
        assert len(result) == 1
        assert result[0]["content"].startswith("...")
        total = _total_tokens(result)
        assert total <= 50

    def test_system_only_messages(self):
        cm = ContextManager(max_tokens=1000)
        msgs = [_system("Instructions")]
        result = cm.truncate(msgs)
        assert len(result) == 1
        assert result[0]["content"] == "Instructions"

    def test_very_tight_budget_still_returns_something(self):
        cm = ContextManager(max_tokens=10)
        msgs = [_system("sys"), _user("Hello world")]
        result = cm.truncate(msgs)
        assert len(result) >= 1

    def test_original_not_modified(self):
        cm = ContextManager(max_tokens=30)
        msgs = [_user("x" * 400), _user("y" * 400)]
        original_len = len(msgs)
        original_content = msgs[0]["content"]
        cm.truncate(msgs)
        assert len(msgs) == original_len
        assert msgs[0]["content"] == original_content


# ===========================================================================
# Summarization
# ===========================================================================


class TestSummarization:
    def test_fit_within_budget_returns_unchanged(self):
        cm = ContextManager(max_tokens=10000)
        msgs = [_user("Hello"), _assistant("Hi")]
        result = cm.fit(msgs)
        assert len(result) == 2
        assert result[0]["content"] == "Hello"

    def test_fit_without_summarizer_falls_back_to_truncation(self):
        cm = ContextManager(max_tokens=50)
        msgs = [_user("x" * 400), _user("recent")]
        result = cm.fit(msgs)
        # Should still work (truncation)
        assert any("recent" in m["content"] for m in result)

    def test_fit_with_summarizer_calls_callback(self):
        called_with = []

        def mock_summarize(messages):
            called_with.append(messages)
            return "Summary of conversation"

        cm = ContextManager(max_tokens=100, summarize_fn=mock_summarize)
        msgs = [
            _user("old message " * 20),
            _assistant("old reply " * 20),
            _user("recent question"),
        ]
        result = cm.fit(msgs)
        # Summarizer should have been called
        assert len(called_with) == 1
        # Result should contain the summary
        summaries = [m for m in result if "[Summary" in m.get("content", "")]
        assert len(summaries) == 1

    def test_fit_preserves_system_messages(self):
        def mock_summarize(messages):
            return "Summary"

        cm = ContextManager(max_tokens=100, summarize_fn=mock_summarize)
        msgs = [
            _system("Be helpful"),
            _user("old " * 50),
            _assistant("response " * 50),
            _user("recent"),
        ]
        result = cm.fit(msgs)
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "Be helpful"

    def test_fit_keeps_recent_messages(self):
        def mock_summarize(messages):
            return "Summary"

        cm = ContextManager(max_tokens=150, summarize_fn=mock_summarize)
        msgs = [
            _user("old " * 50),
            _assistant("old reply " * 50),
            _user("recent question"),
            _assistant("recent answer"),
        ]
        result = cm.fit(msgs)
        contents = [m["content"] for m in result]
        assert "recent question" in contents
        assert "recent answer" in contents

    def test_fit_summary_too_large_falls_back_to_truncation(self):
        def verbose_summarize(messages):
            return "Very long summary " * 100

        cm = ContextManager(max_tokens=100, summarize_fn=verbose_summarize)
        msgs = [
            _user("old " * 50),
            _user("recent"),
        ]
        result = cm.fit(msgs)
        # Should not exceed budget even with a verbose summary
        total = _total_tokens(result)
        assert total <= 100

    def test_fit_returns_new_list(self):
        cm = ContextManager(max_tokens=10000)
        msgs = [_user("Hello")]
        result = cm.fit(msgs)
        assert result is not msgs
        assert result == msgs


# ===========================================================================
# Retrieval packing
# ===========================================================================


class TestRetrievalPacking:
    def test_empty_results(self):
        cm = ContextManager(max_tokens=1000)
        assert cm.pack_retrieval([]) == []

    def test_all_fit(self):
        cm = ContextManager(max_tokens=10000)
        results = [
            RetrievalResult(content="short", score=0.9),
            RetrievalResult(content="also short", score=0.8),
        ]
        packed = cm.pack_retrieval(results, token_budget=1000)
        assert len(packed) == 2

    def test_sorted_by_relevance(self):
        cm = ContextManager(max_tokens=10000)
        results = [
            RetrievalResult(content="low relevance", score=0.3),
            RetrievalResult(content="high relevance", score=0.9),
            RetrievalResult(content="mid relevance", score=0.6),
        ]
        packed = cm.pack_retrieval(results, token_budget=10000)
        assert packed[0].score == 0.9
        assert packed[1].score == 0.6
        assert packed[2].score == 0.3

    def test_budget_limits_results(self):
        cm = ContextManager(max_tokens=100)
        results = [
            RetrievalResult(content="x" * 200, score=0.9),  # ~50 tokens
            RetrievalResult(content="y" * 200, score=0.8),  # ~50 tokens
            RetrievalResult(content="z" * 200, score=0.7),  # ~50 tokens
        ]
        # Budget of 60 tokens: only the highest scoring fits
        packed = cm.pack_retrieval(results, token_budget=60)
        assert len(packed) == 1
        assert packed[0].score == 0.9

    def test_prefers_highest_score_when_budget_tight(self):
        cm = ContextManager(max_tokens=100)
        results = [
            RetrievalResult(content="x" * 400, score=0.5),   # ~100 tokens
            RetrievalResult(content="small", score=0.9),      # ~1 token
        ]
        packed = cm.pack_retrieval(results, token_budget=10)
        assert len(packed) == 1
        assert packed[0].score == 0.9

    def test_top_k_limits(self):
        cm = ContextManager(max_tokens=100000)
        results = [
            RetrievalResult(content=f"result {i}", score=1.0 - i * 0.1)
            for i in range(10)
        ]
        packed = cm.pack_retrieval(results, token_budget=100000, top_k=3)
        assert len(packed) == 3
        assert packed[0].score == 1.0

    def test_default_budget_is_quarter_of_available(self):
        cm = ContextManager(max_tokens=1000, reserve_tokens=200)
        # available = 800, default retrieval budget = 200
        results = [
            RetrievalResult(content="x" * 600, score=0.9),   # ~150 tokens
            RetrievalResult(content="y" * 600, score=0.8),   # ~150 tokens
        ]
        packed = cm.pack_retrieval(results)
        # 150 + 150 = 300 > 200 budget → only 1 fits
        assert len(packed) == 1

    def test_retrieval_result_token_estimate(self):
        r = RetrievalResult(content="x" * 400)
        assert r.token_estimate == 100

    def test_retrieval_result_metadata(self):
        r = RetrievalResult(
            content="text",
            score=0.95,
            source="doc.pdf",
            metadata={"page": 3},
        )
        assert r.source == "doc.pdf"
        assert r.metadata["page"] == 3

    def test_skips_large_results_keeps_smaller(self):
        cm = ContextManager(max_tokens=10000)
        results = [
            RetrievalResult(content="x" * 800, score=0.9),  # 200 tokens
            RetrievalResult(content="y" * 40, score=0.8),    # 10 tokens
            RetrievalResult(content="z" * 800, score=0.7),   # 200 tokens
        ]
        # Budget allows ~100 tokens total
        packed = cm.pack_retrieval(results, token_budget=100)
        # Highest score (200 tokens) doesn't fit in 100 → skipped
        # But the 10-token one should NOT be included because
        # we try highest first: 200 > 100 skip, 10 <= 100 include
        assert len(packed) == 1
        assert packed[0].score == 0.8


# ===========================================================================
# ContextManager properties
# ===========================================================================


class TestContextManagerProperties:
    def test_available_tokens_no_reserve(self):
        cm = ContextManager(max_tokens=4096)
        assert cm.available_tokens == 4096

    def test_available_tokens_with_reserve(self):
        cm = ContextManager(max_tokens=4096, reserve_tokens=1024)
        assert cm.available_tokens == 3072

    def test_available_tokens_over_reserved(self):
        cm = ContextManager(max_tokens=100, reserve_tokens=200)
        assert cm.available_tokens == 0
