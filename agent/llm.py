from __future__ import annotations

from typing import Any

from litellm import completion


MODEL = "ollama/qwen3:8b"


def ask_llm(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
):
    """Send the conversation to the local Ollama model through LiteLLM."""
    kwargs: dict[str, Any] = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.3,
    }
    if tools:
        kwargs["tools"] = tools

    return completion(**kwargs)
