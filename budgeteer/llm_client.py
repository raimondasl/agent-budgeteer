"""LLM client wrapper that captures usage metrics.

Wraps any callable LLM function to track prompt/completion tokens,
latency, and cost transparently.
"""

from __future__ import annotations

import time
from typing import Any, Callable

from budgeteer.models import LLMResponse


class LLMClient:
    """Wraps a user-provided LLM callable to capture usage metrics.

    The callable should accept (model, messages, max_tokens, temperature, **kwargs)
    and return a dict with keys: "content", "usage" (with "prompt_tokens" and
    "completion_tokens"), and "model".

    Example::

        def my_llm(model, messages, max_tokens, temperature, **kwargs):
            response = openai.chat.completions.create(
                model=model, messages=messages, max_tokens=max_tokens,
                temperature=temperature, **kwargs,
            )
            return {
                "content": response.choices[0].message.content,
                "usage": {"prompt_tokens": response.usage.prompt_tokens,
                          "completion_tokens": response.usage.completion_tokens},
                "model": response.model,
            }

        client = LLMClient(my_llm)
        response = client.complete("gpt-4o-mini", [{"role": "user", "content": "Hi"}])
    """

    def __init__(self, call_fn: Callable[..., dict[str, Any]]):
        """Wrap a callable LLM function for metric tracking."""
        self._call_fn = call_fn
        self._call_count: int = 0
        self._total_prompt_tokens: int = 0
        self._total_completion_tokens: int = 0

    def complete(
        self,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> LLMResponse:
        """Call the wrapped LLM and return a standardized response with metrics."""
        start = time.perf_counter()
        raw = self._call_fn(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
        latency_ms = (time.perf_counter() - start) * 1000

        usage = raw.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        self._call_count += 1
        self._total_prompt_tokens += prompt_tokens
        self._total_completion_tokens += completion_tokens

        return LLMResponse(
            content=raw.get("content", ""),
            model=raw.get("model", model),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            raw_response=raw,
        )

    @property
    def call_count(self) -> int:
        """Number of completed LLM calls."""
        return self._call_count

    @property
    def total_prompt_tokens(self) -> int:
        """Cumulative prompt tokens across all calls."""
        return self._total_prompt_tokens

    @property
    def total_completion_tokens(self) -> int:
        """Cumulative completion tokens across all calls."""
        return self._total_completion_tokens
