from __future__ import annotations

import os
from typing import Any

from openai import OpenAI


LM_STUDIO_BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
DEFAULT_MODEL = "qwen/qwen3-8b"
MODEL = os.getenv("LM_STUDIO_MODEL", DEFAULT_MODEL)
THINKING_MODE = os.getenv("LM_STUDIO_THINKING_MODE", "default").strip().lower()

_client = OpenAI(
    base_url=LM_STUDIO_BASE_URL,
    api_key=os.getenv("LM_STUDIO_API_KEY", "lm-studio"),
)


def _prepare_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if THINKING_MODE not in {"think", "no_think"} or not MODEL.lower().startswith("qwen/"):
        return messages

    prepared = [dict(message) for message in messages]
    for index in range(len(prepared) - 1, -1, -1):
        if prepared[index].get("role") != "user":
            continue
        message = prepared[index]
        content = str(message.get("content", ""))
        marker = "/think" if THINKING_MODE == "think" else "/no_think"
        if marker not in content:
            prepared[index] = {**message, "content": f"{content.rstrip()}\n{marker}"}
        break
    return prepared


def _thinking_options() -> dict[str, Any]:
    """Return LM Studio reasoning controls for models that expose them via the API."""
    if THINKING_MODE not in {"think", "no_think"}:
        return {}

    if MODEL.lower().startswith("qwen/"):
        return {}

    # LM Studio exposes a generic reasoning setting for supported reasoning models.
    # Gemma 4's LM Studio model definition maps its Enable Thinking switch to this
    # reasoning control when using the local API.
    return {"extra_body": {"reasoning": "on" if THINKING_MODE == "think" else "off"}}


def ask_llm(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
):
    """Call LM Studio's OpenAI-compatible Chat Completions endpoint."""
    kwargs: dict[str, Any] = {
        "model": MODEL,
        "messages": _prepare_messages(messages),
    }
    kwargs.update(_thinking_options())
    if tools:
        kwargs["tools"] = tools

    return _client.chat.completions.create(**kwargs)
