from types import SimpleNamespace

import pytest

import agent.llm as llm


def test_ask_llm_sets_ollama_generation_limits(monkeypatch) -> None:
    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(llm, "completion", fake_completion)

    llm.ask_llm(
        [{"role": "user", "content": "こんにちは"}],
        tools=[{"type": "function", "function": {"name": "test"}}],
    )

    assert captured["temperature"] == 0.3
    assert captured["max_tokens"] == 512
    assert captured["extra_body"]["options"]["num_predict"] == 512
    assert captured["extra_body"]["options"]["repeat_penalty"] == 1.1
    assert captured["tools"]


def test_ask_llm_retries_once_after_repeat_limit_error(monkeypatch) -> None:
    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise RuntimeError("prediction aborted, token repeat limit reached")
        return SimpleNamespace(ok=True)

    monkeypatch.setattr(llm, "completion", fake_completion)

    result = llm.ask_llm([{"role": "user", "content": "実行してください"}])

    assert result.ok is True
    assert len(calls) == 2
    assert calls[0]["max_tokens"] == 512
    assert calls[0]["temperature"] == 0.3
    assert calls[1]["max_tokens"] == 256
    assert calls[1]["temperature"] == 0.2
    assert calls[1]["extra_body"]["options"]["num_predict"] == 256


def test_ask_llm_does_not_retry_unrelated_error(monkeypatch) -> None:
    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("connection failed")

    monkeypatch.setattr(llm, "completion", fake_completion)

    with pytest.raises(RuntimeError, match="connection failed"):
        llm.ask_llm([{"role": "user", "content": "こんにちは"}])

    assert len(calls) == 1
