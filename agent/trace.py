from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any
from uuid import uuid4


DEFAULT_TRACE_PATH = Path.home() / ".local-ai-agent" / "traces.jsonl"
MAX_TRACE_TEXT = 4_000
MAX_TRACE_FILE_BYTES = 10_000_000


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
            self._rotate_if_needed(len(line.encode("utf-8")) + 1)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except OSError:
            return

    def _rotate_if_needed(self, incoming_bytes: int) -> None:
        try:
            if not self.path.exists():
                return
            if self.path.stat().st_size + incoming_bytes <= MAX_TRACE_FILE_BYTES:
                return
            rotated = self.path.with_suffix(self.path.suffix + ".1")
            if rotated.exists():
                rotated.unlink()
            self.path.replace(rotated)
        except OSError:
            return

    def summary(self) -> dict[str, int | float]:
        metrics: dict[str, int | float] = {
            "llm_calls": 0,
            "tool_calls": 0,
            "llm_duration_ms": 0,
            "tool_duration_ms": 0,
            "prompt_chars": 0,
            "tool_schema_chars": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "reasoning_tokens": 0,
        }
        if not self.enabled or not self.path.exists():
            return metrics

        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return metrics

        for line in lines:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event") == "llm":
                metrics["llm_calls"] += 1
                metrics["llm_duration_ms"] += int(event.get("duration_ms", 0) or 0)
                metrics["prompt_chars"] += int(event.get("prompt_chars", 0) or 0)
                metrics["tool_schema_chars"] += int(event.get("tool_schema_chars", 0) or 0)
                metrics["prompt_tokens"] += int(event.get("prompt_tokens", 0) or 0)
                metrics["completion_tokens"] += int(event.get("completion_tokens", 0) or 0)
                metrics["reasoning_tokens"] += int(event.get("reasoning_tokens", 0) or 0)
            elif event.get("event") == "tool":
                metrics["tool_calls"] += 1
                metrics["tool_duration_ms"] += int(event.get("duration_ms", 0) or 0)

        llm_calls = int(metrics["llm_calls"])
        tool_calls = int(metrics["tool_calls"])
        metrics["avg_llm_duration_ms"] = (
            round(int(metrics["llm_duration_ms"]) / llm_calls, 1)
            if llm_calls else 0
        )
        metrics["avg_tool_duration_ms"] = (
            round(int(metrics["tool_duration_ms"]) / tool_calls, 1)
            if tool_calls else 0
        )
        return metrics

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
