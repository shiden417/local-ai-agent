from __future__ import annotations

from agent import llm


def test_ask_llm_uses_thinking_off_only_as_final_repeat_fallback(monkeypatch) -> None:
    calls = []

    class RepeatError(Exception):
        pass

    def fake_completion(**kwargs):
        calls.append(kwargs)
        if len(calls) < 3:
            raise RepeatError("prediction aborted, token repeat limit reached")
        return {"ok": True}

    monkeypatch.setattr(llm, "completion", fake_completion)

    result = llm.ask_llm([{"role": "user", "content": "test"}])

    assert result == {"ok": True}
    assert len(calls) == 3
    assert "think" not in calls[0]
    assert "think" not in calls[1]
    assert calls[2]["think"] is False
    assert calls[2]["allowed_openai_params"] == ["think"]
