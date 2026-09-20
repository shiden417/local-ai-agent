from __future__ import annotations

from typing import Any

from litellm import completion


MODEL = "ollama/qwen3:8b"
DEFAULT_MAX_TOKENS = 512
RECOVERY_MAX_TOKENS = 256


def _is_repeat_limit_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "token repeat limit reached" in message


def _completion(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    *,
    max_tokens: int,
    temperature: float,
    think: bool | None = None,
):
    kwargs: dict[str, Any] = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "extra_body": {
            "options": {
                "num_predict": max_tokens,
                "repeat_penalty": 1.1,
            }
        },
    }
    if tools:
        kwargs["tools"] = tools
    if think is not None:
        kwargs["think"] = think
        kwargs["allowed_openai_params"] = ["think"]

    return completion(**kwargs)


def ask_llm(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
):
    """Send the conversation to Qwen3 through LiteLLM with repeat protection."""
    try:
        return _completion(
            messages,
            tools,
            max_tokens=DEFAULT_MAX_TOKENS,
            temperature=0.3,
        )
    except Exception as exc:
        if not _is_repeat_limit_error(exc):
            raise

        # Ollama can abort a generation when it enters a repetitive loop.
        # Retry once with a shorter generation budget and slightly stronger
        # repetition suppression before surfacing the provider error.
        try:
            return _completion(
                messages,
                tools,
                max_tokens=RECOVERY_MAX_TOKENS,
                temperature=0.2,
            )
        except Exception as recovery_exc:
            if not _is_repeat_limit_error(recovery_exc):
                raise

            # Final emergency fallback for Qwen3: disable thinking so the
            # bounded recovery budget is spent on a direct response.
            return _completion(
                messages,
                tools,
                max_tokens=RECOVERY_MAX_TOKENS,
                temperature=0.1,
                think=False,
            )
