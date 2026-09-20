from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any
from uuid import uuid4


DEFAULT_TRACE_PATH = Path.home() / ".local-ai-agent" / "traces.jsonl"
MAX_TRACE_TEXT = 4_000


def _compact(value: Any, limit: int = MAX_TRACE_TEXT) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    text = str(value)
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 32)] + "...[truncated]"


def _json_size(value: Any) -> int:
    try:
        return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")))
    except (TypeError, ValueError):
        return len(str(value))


def _usage_value(usage: Any, name: str) -> int | None:
    if usage is None:
        return None
    value = usage.get(name) if isinstance(usage, dict) else getattr(usage, name, None)
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def extract_usage(response: Any) -> dict[str, int | None]:
    usage = getattr(response, "usage", None)
    reasoning = None
    details = (
        usage.get("completion_tokens_details")
        if isinstance(usage, dict)
        else getattr(usage, "completion_tokens_details", None)
    )
    reasoning = _usage_value(details, "reasoning_tokens")
    return {
        "prompt_tokens": _usage_value(usage, "prompt_tokens"),
        "completion_tokens": _usage_value(usage, "completion_tokens"),
        "reasoning_tokens": reasoning,
    }


class TraceRecorder:
    """Local JSONL trace recorder with payloads disabled by default."""

    def __init__(
        self,
        path: str | Path = DEFAULT_TRACE_PATH,
        *,
        enabled: bool | None = None,
        include_payloads: bool | None = None,
    ) -> None:
        self.enabled = (
            os.getenv("JARVIS_TRACE_ENABLED", "0" if os.getenv("PYTEST_CURRENT_TEST") else "1").strip().lower()
            not in {"0", "false", "off", "no"}
            if enabled is None
            else bool(enabled)
        )
        self.include_payloads = (
            os.getenv("JARVIS_TRACE_INCLUDE_PAYLOADS", "0").strip().lower()
            in {"1", "true", "on", "yes"}
            if include_payloads is None
            else bool(include_payloads)
        )
        self.path = Path(path).expanduser().resolve()
        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def new_run_id() -> str:
        return uuid4().hex[:16]

    def record(self, event: str, **fields: Any) -> None:
        if not self.enabled:
            return

        payload = {
            "timestamp": time.time(),
            "event": event,
            **{
                key: _compact(value)
                if key in {"goal", "content", "error", "summary"}
                else value
                for key, value in fields.items()
            },
        }
        try:
            line = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except OSError:
            return

    def run_start(
        self,
        run_id: str,
        *,
        task_id: str,
        mode: str,
        goal: str,
        model: str,
    ) -> None:
        self.record(
            "run_start",
            run_id=run_id,
            task_id=task_id,
            mode=mode,
            goal=goal,
            model=model,
        )

    def llm(
        self,
        run_id: str,
        *,
        iteration: int,
        duration_ms: int,
        prompt_chars: int,
        tool_schema_chars: int,
        response: Any,
    ) -> None:
        fields = {
            "run_id": run_id,
            "iteration": iteration,
            "duration_ms": duration_ms,
            "prompt_chars": prompt_chars,
            "tool_schema_chars": tool_schema_chars,
            **extract_usage(response),
        }
        if self.include_payloads:
            message = response.choices[0].message
            fields["content"] = getattr(message, "content", None)
        self.record("llm", **fields)

    def tool(
        self,
        run_id: str,
        *,
        iteration: int,
        name: str,
        arguments: dict[str, Any],
        result: dict[str, Any],
        duration_ms: int,
        safety_decision: str,
    ) -> None:
        fields: dict[str, Any] = {
            "run_id": run_id,
            "iteration": iteration,
            "tool": name,
            "duration_ms": duration_ms,
            "ok": bool(result.get("ok")),
            "result_chars": _json_size(result),
            "safety": safety_decision,
        }
        if self.include_payloads:
            fields["arguments"] = arguments
            fields["result"] = result
        self.record("tool", **fields)

    def run_end(
        self,
        run_id: str,
        *,
        task_id: str,
        status: str,
        iterations: int,
        tool_calls: int,
    ) -> None:
        self.record(
            "run_end",
            run_id=run_id,
            task_id=task_id,
            status=status,
            iterations=iterations,
            tool_calls=tool_calls,
        )


__all__ = ["TraceRecorder", "extract_usage"]
