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
    assert "extra_body" not in captured


@pytest.mark.parametrize(
    ("mode", "expected_reasoning"),
    [
        ("think", "on"),
        ("no_think", "off"),
    ],
)
def test_gemma_thinking_mode_uses_lm_studio_reasoning_control(
    monkeypatch, mode: str, expected_reasoning: str
) -> None:
    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(ok=True)

    monkeypatch.setattr(llm._client.chat.completions, "create", fake_create)
    monkeypatch.setattr(llm, "MODEL", "google/gemma-4-e4b")
    monkeypatch.setattr(llm, "THINKING_MODE", mode)

    llm.ask_llm([{"role": "user", "content": "調査してください"}])

    assert captured["extra_body"] == {"reasoning": expected_reasoning}
    assert captured["messages"] == [{"role": "user", "content": "調査してください"}]


def test_non_thinking_default_does_not_send_reasoning_control(monkeypatch) -> None:
    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(ok=True)

    monkeypatch.setattr(llm._client.chat.completions, "create", fake_create)
    monkeypatch.setattr(llm, "MODEL", "google/gemma-4-e4b")
    monkeypatch.setattr(llm, "THINKING_MODE", "default")

    llm.ask_llm([{"role": "user", "content": "調査してください"}])

    assert "extra_body" not in captured
