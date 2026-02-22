"""Tests for budgeteer.llm_client."""

import pytest

from budgeteer.llm_client import LLMClient


def _make_callable(content="Hello", prompt_tokens=10, completion_tokens=5, model=None):
    """Return a fake LLM callable that returns a standard response dict."""
    def fn(*, model: str, messages, max_tokens, temperature, **kwargs):
        resp = {
            "content": content,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
        }
        if model is not None:
            resp["model"] = model
        return resp
    return fn


class TestLLMClientBasics:
    def test_complete_returns_llm_response(self):
        client = LLMClient(_make_callable(model="gpt-4o"))
        resp = client.complete("gpt-4o", [{"role": "user", "content": "Hi"}])
        assert resp.content == "Hello"
        assert resp.model == "gpt-4o"
        assert resp.prompt_tokens == 10
        assert resp.completion_tokens == 5
        assert resp.latency_ms >= 0
        assert resp.raw_response is not None

    def test_call_count_increments(self):
        client = LLMClient(_make_callable())
        assert client.call_count == 0
        for _ in range(3):
            client.complete("m", [])
        assert client.call_count == 3

    def test_token_totals_accumulate(self):
        client = LLMClient(_make_callable(prompt_tokens=100, completion_tokens=50))
        client.complete("m", [])
        client.complete("m", [])
        assert client.total_prompt_tokens == 200
        assert client.total_completion_tokens == 100

    def test_passes_model_and_messages(self):
        received = {}

        def fn(*, model, messages, max_tokens, temperature, **kwargs):
            received["model"] = model
            received["messages"] = messages
            return {"content": "", "usage": {}, "model": model}

        client = LLMClient(fn)
        msgs = [{"role": "user", "content": "test"}]
        client.complete("gpt-4o-mini", msgs)
        assert received["model"] == "gpt-4o-mini"
        assert received["messages"] is msgs

    def test_passes_max_tokens_and_temperature(self):
        received = {}

        def fn(*, model, messages, max_tokens, temperature, **kwargs):
            received["max_tokens"] = max_tokens
            received["temperature"] = temperature
            return {"content": "", "usage": {}}

        client = LLMClient(fn)
        client.complete("m", [], max_tokens=2048, temperature=0.3)
        assert received["max_tokens"] == 2048
        assert received["temperature"] == 0.3

    def test_passes_extra_kwargs(self):
        received = {}

        def fn(*, model, messages, max_tokens, temperature, **kwargs):
            received.update(kwargs)
            return {"content": "", "usage": {}}

        client = LLMClient(fn)
        client.complete("m", [], top_p=0.9, stop=["END"])
        assert received["top_p"] == 0.9
        assert received["stop"] == ["END"]


class TestLLMClientMetrics:
    def test_latency_is_positive(self):
        client = LLMClient(_make_callable())
        resp = client.complete("m", [])
        assert resp.latency_ms > 0

    def test_raw_response_preserved(self):
        raw = {"content": "ok", "usage": {"prompt_tokens": 1, "completion_tokens": 1}, "model": "m", "extra": 42}

        def fn(*, model, messages, max_tokens, temperature, **kwargs):
            return raw

        client = LLMClient(fn)
        resp = client.complete("m", [])
        assert resp.raw_response is raw
        assert resp.raw_response["extra"] == 42


class TestLLMClientEdgeCases:
    def test_missing_usage_defaults_to_zero(self):
        def fn(*, model, messages, max_tokens, temperature, **kwargs):
            return {"content": "hi"}

        client = LLMClient(fn)
        resp = client.complete("m", [])
        assert resp.prompt_tokens == 0
        assert resp.completion_tokens == 0
        assert client.total_prompt_tokens == 0
        assert client.total_completion_tokens == 0

    def test_missing_content_defaults_to_empty(self):
        def fn(*, model, messages, max_tokens, temperature, **kwargs):
            return {"usage": {"prompt_tokens": 5, "completion_tokens": 3}}

        client = LLMClient(fn)
        resp = client.complete("m", [])
        assert resp.content == ""

    def test_missing_model_falls_back_to_input(self):
        def fn(*, model, messages, max_tokens, temperature, **kwargs):
            return {"content": "hi", "usage": {}}

        client = LLMClient(fn)
        resp = client.complete("my-model", [])
        assert resp.model == "my-model"

    def test_callable_raises_propagates(self):
        def fn(*, model, messages, max_tokens, temperature, **kwargs):
            raise RuntimeError("LLM failed")

        client = LLMClient(fn)
        with pytest.raises(RuntimeError, match="LLM failed"):
            client.complete("m", [])
        # Counters should not increment on failure
        assert client.call_count == 0
        assert client.total_prompt_tokens == 0
