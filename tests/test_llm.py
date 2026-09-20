from __future__ import annotations

from types import SimpleNamespace

import pytest

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
    assert captured == {
        "model": llm.MODEL,
        "messages": [{"role": "user", "content": "こんにちは"}],
        "tools": [{"type": "function", "function": {"name": "test"}}],
    }


def test_ask_llm_omits_tools_when_not_provided(monkeypatch) -> None:
    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(ok=True)

    monkeypatch.setattr(llm._client.chat.completions, "create", fake_create)

    llm.ask_llm([{"role": "user", "content": "こんにちは"}])

    assert captured == {
        "model": llm.MODEL,
        "messages": [{"role": "user", "content": "こんにちは"}],
    }


def test_ask_llm_propagates_provider_errors(monkeypatch) -> None:
    calls = []

    def fake_create(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("connection failed")

    monkeypatch.setattr(llm._client.chat.completions, "create", fake_create)

    with pytest.raises(RuntimeError, match="connection failed"):
        llm.ask_llm([{"role": "user", "content": "こんにちは"}])

    assert len(calls) == 1


def test_qwen_no_think_mode_marks_latest_user_message(monkeypatch) -> None:
    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(ok=True)

    monkeypatch.setattr(llm._client.chat.completions, "create", fake_create)
    monkeypatch.setattr(llm, "MODEL", "qwen/qwen3-8b")
    monkeypatch.setattr(llm, "THINKING_MODE", "no_think")

    llm.ask_llm([{"role": "user", "content": "調査してください"}])

    assert captured["messages"][-1]["content"].endswith("/no_think")
