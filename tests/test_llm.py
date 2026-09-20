from __future__ import annotations

from types import SimpleNamespace

from agent import llm


def test_ask_llm_uses_lm_studio_openai_compatible_request(monkeypatch) -> None:
    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(ok=True)

    monkeypatch.setattr(llm._client.chat.completions, "create", fake_create)

    result = llm.ask_llm(
        [{"role": "user", "content": "こんにちは"}],
        tools=[{"type": "function", "function": {"name": "test"}}],
    )

    assert result.ok is True
    assert captured["model"] == llm.MODEL
    assert captured["temperature"] == llm.TEMPERATURE
    assert captured["max_tokens"] == llm.DEFAULT_MAX_TOKENS
    assert captured["tools"]


def test_ask_llm_retries_after_recoverable_generation_error(monkeypatch) -> None:
    calls = []

    def fake_create(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise RuntimeError("prediction aborted, token repeat limit reached")
        return SimpleNamespace(ok=True)

    monkeypatch.setattr(llm._client.chat.completions, "create", fake_create)

    result = llm.ask_llm([{"role": "user", "content": "実行してください"}])

    assert result.ok is True
    assert len(calls) == 2
    assert calls[0]["max_tokens"] == llm.DEFAULT_MAX_TOKENS
    assert calls[1]["max_tokens"] == llm.RECOVERY_MAX_TOKENS


def test_ask_llm_does_not_retry_unrelated_error(monkeypatch) -> None:
    calls = []

    def fake_create(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("connection failed")

    monkeypatch.setattr(llm._client.chat.completions, "create", fake_create)

    import pytest

    with pytest.raises(RuntimeError, match="connection failed"):
        llm.ask_llm([{"role": "user", "content": "こんにちは"}])

    assert len(calls) == 1
