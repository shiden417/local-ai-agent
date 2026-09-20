from __future__ import annotations

import os
from typing import Any

from openai import OpenAI


LM_STUDIO_BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
MODEL = os.getenv("LM_STUDIO_MODEL", "qwen/qwen3-8b")
DEFAULT_MAX_TOKENS = int(os.getenv("JARVIS_MAX_TOKENS", "1024"))
RECOVERY_MAX_TOKENS = int(os.getenv("JARVIS_RECOVERY_MAX_TOKENS", "512"))
TEMPERATURE = float(os.getenv("JARVIS_TEMPERATURE", "0.3"))

_client = OpenAI(
    base_url=LM_STUDIO_BASE_URL,
    api_key=os.getenv("LM_STUDIO_API_KEY", "lm-studio"),
)


def _is_recoverable_generation_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "token repeat limit",
            "repetitive",
            "generation aborted",
            "prediction aborted",
        )
    )


def _completion(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    *,
    max_tokens: int,
    temperature: float,
):
    kwargs: dict[str, Any] = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if tools:
        kwargs["tools"] = tools

    return _client.chat.completions.create(**kwargs)


def ask_llm(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
):
    """Send a request to Qwen through LM Studio's OpenAI-compatible API."""
    try:
        return _completion(
            messages,
            tools,
            max_tokens=DEFAULT_MAX_TOKENS,
            temperature=TEMPERATURE,
        )
    except Exception as exc:
        if not _is_recoverable_generation_error(exc):
            raise

        try:
            return _completion(
                messages,
                tools,
                max_tokens=RECOVERY_MAX_TOKENS,
                temperature=max(0.1, TEMPERATURE - 0.1),
            )
        except Exception as recovery_exc:
            if not _is_recoverable_generation_error(recovery_exc):
                raise

            return _completion(
                messages,
                tools,
                max_tokens=RECOVERY_MAX_TOKENS,
                temperature=0.1,
            )
