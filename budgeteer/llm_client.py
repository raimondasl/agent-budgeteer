"""LLM client wrapper that captures usage metrics.

Wraps any callable LLM function to track prompt/completion tokens,
latency, and cost transparently.
"""

from __future__ import annotations

import time
from typing import Any, Callable

from budgeteer.models import LLMResponse, ModelTier, compute_cost


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

    def __init__(
        self,
        call_fn: Callable[..., dict[str, Any]],
        model_tiers: list[ModelTier] | None = None,
    ):
        """Wrap a callable LLM function for metric tracking."""
        self._call_fn = call_fn
        self._model_tiers = model_tiers or []
        self._call_count: int = 0
        self._total_prompt_tokens: int = 0
        self._total_completion_tokens: int = 0
        self._total_cost_usd: float = 0.0

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

        actual_model = raw.get("model", model)
        cost = compute_cost(actual_model, prompt_tokens, completion_tokens, self._model_tiers)

        self._call_count += 1
        self._total_prompt_tokens += prompt_tokens
        self._total_completion_tokens += completion_tokens
        self._total_cost_usd += cost

        return LLMResponse(
            content=raw.get("content", ""),
            model=actual_model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            cost_usd=cost,
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

    @property
    def total_cost_usd(self) -> float:
        """Cumulative cost in USD across all calls."""
        return self._total_cost_usd

    def complete_stream(
        self,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs: Any,
    ):
        """Call the wrapped LLM in streaming mode, yielding chunks.

        The callable should return an iterable of dicts with "content" keys
        for intermediate chunks, with the final chunk containing "usage"
        and "model". Returns an ``LLMResponse`` as the generator's return
        value (accessible via ``StopIteration.value``).

        Yields:
            dict: Intermediate chunks with "content" key.
        """
        start = time.perf_counter()
        raw_stream = self._call_fn(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            **kwargs,
        )

        full_content = ""
        last_chunk = None
        for chunk in raw_stream:
            full_content += chunk.get("content", "")
            last_chunk = chunk
            yield chunk

        latency_ms = (time.perf_counter() - start) * 1000

        # Extract final usage from last chunk or defaults
        usage = (last_chunk or {}).get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        actual_model = (last_chunk or {}).get("model", model)

        cost = compute_cost(actual_model, prompt_tokens, completion_tokens, self._model_tiers)

        self._call_count += 1
        self._total_prompt_tokens += prompt_tokens
        self._total_completion_tokens += completion_tokens
        self._total_cost_usd += cost

        return LLMResponse(
            content=full_content,
            model=actual_model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            cost_usd=cost,
            raw_response=last_chunk,
        )
