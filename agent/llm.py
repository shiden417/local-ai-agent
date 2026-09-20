from __future__ import annotations

import os
from typing import Any

from openai import OpenAI


LM_STUDIO_BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
DEFAULT_MODEL = "qwen/qwen3-8b"
MODEL = os.getenv("LM_STUDIO_MODEL", DEFAULT_MODEL)

_client = OpenAI(
    base_url=LM_STUDIO_BASE_URL,
    api_key=os.getenv("LM_STUDIO_API_KEY", "lm-studio"),
)


def ask_llm(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
):
    """Call LM Studio's OpenAI-compatible Chat Completions endpoint."""
    kwargs: dict[str, Any] = {
        "model": MODEL,
        "messages": messages,
    }
    if tools:
        kwargs["tools"] = tools

    return _client.chat.completions.create(**kwargs)
