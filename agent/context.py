from __future__ import annotations

import json
from typing import Any


DEFAULT_MAX_CHARS = 48_000
DEFAULT_KEEP_RECENT_BLOCKS = 4
SUMMARY_MAX_CHARS = 6_000


class ContextManager:
    """Keep LLM context bounded without breaking tool-call message sequences."""

    def __init__(
        self,
        max_chars: int = DEFAULT_MAX_CHARS,
        keep_recent_blocks: int = DEFAULT_KEEP_RECENT_BLOCKS,
    ) -> None:
        if max_chars <= 0:
            raise ValueError("max_chars must be positive")
        if keep_recent_blocks < 1:
            raise ValueError("keep_recent_blocks must be at least 1")

        self.max_chars = max_chars
        self.keep_recent_blocks = keep_recent_blocks

    def prepare(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self._estimate_chars(messages) <= self.max_chars:
            return list(messages)

        system_messages = [
            message for message in messages if message.get("role") == "system"
        ]
        conversational = [
            message for message in messages if message.get("role") != "system"
        ]

        blocks = self._split_blocks(conversational)

        if len(blocks) <= self.keep_recent_blocks:
            return self._trim_tool_messages(system_messages + conversational)

        old_blocks = blocks[:-self.keep_recent_blocks]
        recent_blocks = blocks[-self.keep_recent_blocks:]
        old_messages = [message for block in old_blocks for message in block]
        recent_messages = [message for block in recent_blocks for message in block]

        summary = self._build_summary(old_messages)
        compacted = system_messages + [
            {
                "role": "system",
                "content": (
                    "Compacted history summary. This is historical execution "
                    "context; prefer recent messages when they conflict.\n"
                    + summary
                ),
            },
            *recent_messages,
        ]

        return self._trim_tool_messages(compacted)

    def _split_blocks(
        self,
        messages: list[dict[str, Any]],
    ) -> list[list[dict[str, Any]]]:
        blocks: list[list[dict[str, Any]]] = []

        for message in messages:
            role = message.get("role")

            if role in {"user", "assistant"}:
                blocks.append([message])
            elif role == "tool" and blocks:
                blocks[-1].append(message)
            else:
                blocks.append([message])

        return blocks

    def _build_summary(self, messages: list[dict[str, Any]]) -> str:
        lines: list[str] = []

        for message in messages:
            role = message.get("role", "")
            content = message.get("content") or ""

            if role == "user":
                text = _single_line(str(content), 500)
                if text:
                    lines.append(f"user: {text}")
            elif role == "assistant":
                tool_calls = message.get("tool_calls") or []
                if tool_calls:
                    names = [
                        _tool_name(call)
                        for call in tool_calls
                        if _tool_name(call)
                    ]
                    if names:
                        lines.append("assistant tools: " + ", ".join(names))
                elif content:
                    lines.append(f"assistant: {_single_line(str(content), 500)}")
            elif role == "tool":
                try:
                    result = json.loads(str(content))
                except json.JSONDecodeError:
                    result = None

                if isinstance(result, dict):
                    name = message.get("name", "tool")
                    parts = [name]
                    if "ok" in result:
                        parts.append(f"ok={result['ok']}")
                    if "exit_code" in result:
                        parts.append(f"exit_code={result['exit_code']}")
                    if "error" in result:
                        parts.append(
                            f"error={_single_line(str(result['error']), 300)}"
                        )
                    if "path" in result:
                        parts.append(f"path={result['path']}")
                    lines.append("tool: " + ", ".join(parts))
                else:
                    lines.append(
                        "tool: " + _single_line(str(content), 300)
                    )

            if sum(len(line) + 1 for line in lines) >= SUMMARY_MAX_CHARS:
                break

        return "\n".join(lines) or "No older execution history was retained."

    def _trim_tool_messages(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        result = [dict(message) for message in messages]

        while self._estimate_chars(result) > self.max_chars:
            candidate = next(
                (
                    index
                    for index, message in enumerate(result)
                    if message.get("role") == "tool"
                    and len(str(message.get("content", ""))) > 1_000
                ),
                None,
            )
            if candidate is None:
                break

            message = result[candidate]
            content = str(message.get("content", ""))
            result[candidate] = {
                **message,
                "content": _bounded_text(content, 1_000),
            }

        return result

    @staticmethod
    def _estimate_chars(messages: list[dict[str, Any]]) -> int:
        return sum(
            len(str(message.get("content", ""))) + 200
            for message in messages
        )


def _tool_name(tool_call: Any) -> str:
    if isinstance(tool_call, dict):
        function = tool_call.get("function", {})
        if isinstance(function, dict):
            return str(function.get("name", ""))
        return ""

    function = getattr(tool_call, "function", None)
    return str(getattr(function, "name", ""))


def _single_line(value: str, max_chars: int) -> str:
    return " ".join(value.split())[:max_chars]


def _bounded_text(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value

    marker = "\n...[history detail omitted]...\n"
    available = max_chars - len(marker)
    head = max(1, available // 2)
    tail = max(1, available - head)
    return value[:head] + marker + value[-tail:]
