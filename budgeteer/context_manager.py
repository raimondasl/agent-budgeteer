"""Context manager for budget-aware message truncation, summarization, and retrieval packing.

Fits conversation messages and retrieval results within a token budget
by applying a cascade of compression strategies:

1. **Truncation** — drop oldest non-system messages until the context fits.
2. **Summarization** — replace a block of older messages with a summary
   produced by a user-supplied callback (e.g. a cheap LLM call).
3. **Retrieval packing** — select the most relevant retrieval snippets
   that fit within the remaining token budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


# ------------------------------------------------------------------
# Token estimation
# ------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """Rough token count: ~4 characters per token for English text.

    Consistent with the estimator used in :mod:`budgeteer.router`.
    """
    return max(1, len(text) // 4)


def _message_tokens(message: dict[str, Any]) -> int:
    """Estimate tokens for a single chat message (role + content)."""
    content = str(message.get("content", ""))
    role = str(message.get("role", ""))
    # Small overhead for role, formatting, and message delimiter
    return estimate_tokens(role + content) + 4


# ------------------------------------------------------------------
# Retrieval result
# ------------------------------------------------------------------


@dataclass
class RetrievalResult:
    """A single retrieval snippet with relevance score.

    Attributes:
        content: The text content of the snippet.
        score: Relevance score (higher = more relevant).
        source: Optional source identifier / URL.
        metadata: Arbitrary metadata attached to the result.
    """

    content: str
    score: float = 1.0
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def token_estimate(self) -> int:
        return estimate_tokens(self.content)


# ------------------------------------------------------------------
# Context manager
# ------------------------------------------------------------------


#: Type alias for a summarization callback.
#: Takes a list of messages and returns a single summary string.
SummarizeFn = Callable[[list[dict[str, Any]]], str]


class ContextManager:
    """Budget-aware context compression for agent conversations.

    Usage::

        cm = ContextManager(max_tokens=4096)

        # Basic truncation
        trimmed = cm.truncate(messages)

        # Summarization (requires a callback)
        cm_with_summary = ContextManager(
            max_tokens=4096,
            summarize_fn=my_llm_summarize,
        )
        compressed = cm_with_summary.fit(messages)

        # Retrieval packing
        packed = cm.pack_retrieval(results, token_budget=1000)

    Args:
        max_tokens: Maximum token budget for the context window.
        summarize_fn: Optional callback ``(messages) -> summary_text``
            used to compress older messages.  When not provided,
            :meth:`fit` falls back to truncation only.
        reserve_tokens: Tokens to reserve for the model's response
            (subtracted from ``max_tokens`` during fitting).
    """

    def __init__(
        self,
        max_tokens: int = 8192,
        summarize_fn: SummarizeFn | None = None,
        reserve_tokens: int = 0,
    ) -> None:
        self._max_tokens = max_tokens
        self._summarize_fn = summarize_fn
        self._reserve_tokens = reserve_tokens

    @property
    def available_tokens(self) -> int:
        """Effective token budget after reserving response tokens."""
        return max(0, self._max_tokens - self._reserve_tokens)

    # ------------------------------------------------------------------
    # 1. Truncation
    # ------------------------------------------------------------------

    def truncate(
        self, messages: list[dict[str, Any]], max_tokens: int | None = None
    ) -> list[dict[str, Any]]:
        """Truncate messages to fit within *max_tokens*.

        Strategy:
        - System messages (role="system") at the start are always kept.
        - The most recent user/assistant messages are prioritised.
        - Oldest non-system messages are dropped first.
        - If a single message still exceeds the budget, its content
          is truncated from the beginning (keeping the end).

        Returns a new list; the original is not modified.
        """
        if not messages:
            return []

        budget = max_tokens if max_tokens is not None else self.available_tokens

        # Separate leading system messages from the rest
        system_msgs: list[dict[str, Any]] = []
        rest: list[dict[str, Any]] = []
        in_system_prefix = True
        for msg in messages:
            if in_system_prefix and msg.get("role") == "system":
                system_msgs.append(msg)
            else:
                in_system_prefix = False
                rest.append(msg)

        system_cost = sum(_message_tokens(m) for m in system_msgs)

        # If system messages alone exceed budget, truncate the last system message
        if system_cost > budget:
            if system_msgs:
                system_msgs = [self._truncate_message(system_msgs[-1], budget)]
            return system_msgs

        remaining = budget - system_cost

        # Keep messages from the end (most recent) as long as they fit
        kept: list[dict[str, Any]] = []
        for msg in reversed(rest):
            cost = _message_tokens(msg)
            if cost <= remaining:
                kept.append(msg)
                remaining -= cost
            elif remaining > 4:
                # Partially include this message (truncate content)
                kept.append(self._truncate_message(msg, remaining))
                remaining = 0
                break
            else:
                break

        kept.reverse()
        return system_msgs + kept

    @staticmethod
    def _truncate_message(
        message: dict[str, Any], token_budget: int
    ) -> dict[str, Any]:
        """Truncate a single message's content to fit *token_budget*.

        Keeps the end of the content (most recent/relevant context).
        """
        content = str(message.get("content", ""))
        role = str(message.get("role", ""))
        # _message_tokens = estimate_tokens(role + content) + 4
        # We need: len(role + content) // 4 + 4 <= budget
        # So: len(content) <= (budget - 4) * 4 - len(role)
        max_content_chars = max(0, (token_budget - 4) * 4 - len(role))
        if len(content) > max_content_chars:
            keep = max(0, max_content_chars - 3)  # room for "..."
            content = "..." + content[len(content) - keep :] if keep > 0 else "..."
        return {**message, "content": content}

    # ------------------------------------------------------------------
    # 2. Summarization
    # ------------------------------------------------------------------

    def fit(
        self, messages: list[dict[str, Any]], max_tokens: int | None = None
    ) -> list[dict[str, Any]]:
        """Fit messages within the token budget, using summarization if available.

        When a ``summarize_fn`` was provided and the messages exceed the
        budget, older messages are grouped and replaced with a summary.
        Falls back to :meth:`truncate` when no summarizer is configured
        or when the messages already fit.

        Returns a new list; the original is not modified.
        """
        budget = max_tokens if max_tokens is not None else self.available_tokens
        total = sum(_message_tokens(m) for m in messages)

        # Already fits — return as-is
        if total <= budget:
            return list(messages)

        # No summarizer — fall back to truncation
        if self._summarize_fn is None:
            return self.truncate(messages, budget)

        # Separate leading system messages
        system_msgs: list[dict[str, Any]] = []
        rest: list[dict[str, Any]] = []
        in_system = True
        for msg in messages:
            if in_system and msg.get("role") == "system":
                system_msgs.append(msg)
            else:
                in_system = False
                rest.append(msg)

        if not rest:
            return self.truncate(messages, budget)

        system_cost = sum(_message_tokens(m) for m in system_msgs)
        remaining = budget - system_cost

        # Binary search for the split point: summarize messages[:split],
        # keep messages[split:] verbatim, such that summary + kept fits.
        # Start by trying to keep as many recent messages as possible.
        kept_cost = 0
        split = len(rest)
        for i in range(len(rest) - 1, -1, -1):
            cost = _message_tokens(rest[i])
            if kept_cost + cost <= remaining * 0.7:
                # Reserve ~70% for recent messages, ~30% for summary
                kept_cost += cost
                split = i
            else:
                break

        # Ensure we have something to summarize
        if split == 0:
            return self.truncate(messages, budget)

        to_summarize = rest[:split]
        to_keep = rest[split:]

        summary_text = self._summarize_fn(to_summarize)
        summary_msg: dict[str, Any] = {
            "role": "system",
            "content": f"[Summary of earlier conversation]\n{summary_text}",
        }

        result = system_msgs + [summary_msg] + to_keep
        result_cost = sum(_message_tokens(m) for m in result)

        # If summary still too large, truncate as final fallback
        if result_cost > budget:
            return self.truncate(result, budget)

        return result

    # ------------------------------------------------------------------
    # 3. Retrieval packing
    # ------------------------------------------------------------------

    def pack_retrieval(
        self,
        results: list[RetrievalResult],
        token_budget: int | None = None,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Select retrieval results that fit within the token budget.

        Results are sorted by relevance score (highest first) and packed
        greedily until the budget is exhausted.

        Args:
            results: Candidate retrieval results.
            token_budget: Maximum tokens for retrieval content.
                Defaults to 25% of :attr:`available_tokens`.
            top_k: Optional hard limit on number of results.

        Returns a new list of selected results, sorted by score (descending).
        """
        if not results:
            return []

        budget = token_budget if token_budget is not None else self.available_tokens // 4

        # Sort by score descending
        ranked = sorted(results, key=lambda r: r.score, reverse=True)

        if top_k is not None:
            ranked = ranked[:top_k]

        packed: list[RetrievalResult] = []
        used = 0
        for result in ranked:
            cost = result.token_estimate
            if used + cost <= budget:
                packed.append(result)
                used += cost
            # Skip results that don't fit (greedy, no backtracking)

        return packed
