from __future__ import annotations

import json
from dataclasses import dataclass, field
import re
from typing import Any
from urllib.parse import urlparse

from agent.observation import truncate_text


DEFAULT_MAX_TASK_CHARS = 48_000
DEFAULT_KEEP_RECENT_BLOCKS = 4
TASK_SUMMARY_MAX_CHARS = 6_000
DEFAULT_CONVERSATION_MAX_TURNS = 8
DEFAULT_CONVERSATION_MAX_CHARS = 8_000

MAX_TOPIC_CHARS = 300
MAX_GOAL_CHARS = 500
MAX_ANSWER_CHARS = 4_000
MAX_FACTS = 6
MAX_FACT_CHARS = 320
MAX_REFERENCES = 6


@dataclass
class SessionManager:
    """Own task-context compaction, short conversation history, and session carry-over."""

    max_task_chars: int = DEFAULT_MAX_TASK_CHARS
    keep_recent_blocks: int = DEFAULT_KEEP_RECENT_BLOCKS
    max_conversation_turns: int = DEFAULT_CONVERSATION_MAX_TURNS
    max_conversation_chars: int = DEFAULT_CONVERSATION_MAX_CHARS
    current_topic: str = ""
    anchor_goal: str = ""
    last_goal: str = ""
    last_answer: str = ""
    extracted_facts: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)
    _conversation_messages: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.max_task_chars <= 0:
            raise ValueError("max_task_chars must be positive")
        if self.keep_recent_blocks < 1:
            raise ValueError("keep_recent_blocks must be at least 1")
        if self.max_conversation_turns < 1:
            raise ValueError("max_conversation_turns must be at least 1")
        if self.max_conversation_chars < 1:
            raise ValueError("max_conversation_chars must be positive")

    @property
    def has_context(self) -> bool:
        return bool(
            self.current_topic
            or self.last_goal
            or self.last_answer
            or self.extracted_facts
            or self.references
        )

    def recent_conversation_messages(self) -> list[dict[str, Any]]:
        return [dict(message) for message in self._conversation_messages]

    def add_conversation_turn(self, user_content: str, assistant_content: str) -> None:
        user = user_content.strip()
        assistant = assistant_content.strip()
        if not user or not assistant:
            return

        self._conversation_messages.extend(
            [
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ]
        )
        self._trim_conversation()

    def prepare_task_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Bound task history without breaking assistant-tool/tool-result sequences."""
        if self._estimate_chars(messages) <= self.max_task_chars:
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

    def prompt_block(self) -> str:
        if not self.current_topic and not self.last_answer:
            return (
                "[Session Context]\n"
                "No previous task context is retained for this session."
            )

        lines = ["[Session Context]"]
        lines.append(
            "This is carry-over context, not proof of the current task. "
            "Use current-task observations and current user instructions as the source of truth."
        )
        if self.current_topic:
            lines.append(f"Current topic: {self.current_topic}")
        if self.anchor_goal:
            lines.append(f"Topic anchor: {self.anchor_goal}")
        if self.last_goal:
            lines.append(f"Previous goal: {self.last_goal}")
        if self.extracted_facts:
            lines.append("Important retained facts:")
            lines.extend(f"- {fact}" for fact in self.extracted_facts)
        if self.last_answer:
            answer, _ = truncate_text(self.last_answer, 2_000)
            lines.append(
                "Previous answer (reference only; do not claim it was verified in the current task): "
                f"{answer}"
            )
        if self.references:
            lines.append("Relevant references:")
            lines.extend(f"- {url}" for url in self.references)
        return "\n".join(lines)

    def remember_task(
        self,
        goal: str,
        answer: str,
        messages: list[dict[str, Any]],
    ) -> None:
        previous_topic = self.current_topic
        previous_anchor = self.anchor_goal
        previous_facts = list(self.extracted_facts)
        previous_references = list(self.references)
        carry_previous = _is_continuation(goal, previous_topic)

        self.current_topic, _ = truncate_text(str(goal).strip(), MAX_TOPIC_CHARS)
        anchor_source = (
            previous_anchor if carry_previous and previous_anchor else goal
        )
        self.anchor_goal, _ = truncate_text(
            str(anchor_source).strip(),
            MAX_GOAL_CHARS,
        )
        self.last_goal, _ = truncate_text(str(goal).strip(), MAX_GOAL_CHARS)
        self.last_answer, _ = truncate_text(str(answer).strip(), MAX_ANSWER_CHARS)

        facts: list[str] = []
        references: list[str] = []

        for message in reversed(messages):
            if message.get("role") != "tool":
                continue
            raw = str(message.get("content", ""))
            try:
                result = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(result, dict) or not result.get("ok"):
                continue

            name = str(message.get("name", ""))
            if name == "search_web":
                for item in result.get("results", [])[:3]:
                    if not isinstance(item, dict):
                        continue
                    title = str(item.get("title", "")).strip()
                    snippet = str(item.get("snippet", "")).strip()
                    if title or snippet:
                        fact = f"{title}: {snippet}".strip(": ")
                        fact, _ = truncate_text(fact, MAX_FACT_CHARS)
                        facts.append(fact)
                    url = str(item.get("url", "")).strip()
                    if _is_http_url(url):
                        references.append(url)

            elif name == "fetch_web_page":
                title = str(result.get("title", "")).strip()
                content = str(result.get("content", "")).strip()
                if title:
                    facts.append(f"Web page: {title}")
                if content:
                    first_sentences = " ".join(content.split())[:MAX_FACT_CHARS]
                    if first_sentences:
                        facts.append(f"Web content: {first_sentences}")
                url = str(result.get("final_url") or result.get("url", "")).strip()
                if _is_http_url(url):
                    references.append(url)

            elif name in {"file_mutation", "create_file", "edit_file", "delete_file"}:
                path = str(result.get("path", "")).strip()
                if path:
                    facts.append(f"Local file result: {path}")

        combined_facts = facts + (previous_facts if carry_previous else [])
        combined_references = references + (
            previous_references if carry_previous else []
        )
        self.extracted_facts = _dedupe(combined_facts)[:MAX_FACTS]
        self.references = _dedupe(combined_references)[:MAX_REFERENCES]

    def clear(self) -> None:
        self.current_topic = ""
        self.anchor_goal = ""
        self.last_goal = ""
        self.last_answer = ""
        self.extracted_facts.clear()
        self.references.clear()
        self._conversation_messages.clear()

    def _trim_conversation(self) -> None:
        while len(self._conversation_messages) > self.max_conversation_turns * 2:
            self._conversation_messages.pop(0)

        while (
            sum(
                len(str(message.get("content", ""))) + 40
                for message in self._conversation_messages
            ) > self.max_conversation_chars
            and len(self._conversation_messages) > 2
        ):
            self._conversation_messages.pop(0)
            self._conversation_messages.pop(0)

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
                value = _single_line(str(content), 500)
                if value:
                    lines.append(f"user: {value}")
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
                    lines.append("tool: " + _single_line(str(content), 300))

            if sum(len(line) + 1 for line in lines) >= TASK_SUMMARY_MAX_CHARS:
                break

        return "\n".join(lines) or "No older execution history was retained."

    def _trim_tool_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        result = [dict(message) for message in messages]

        while self._estimate_chars(result) > self.max_task_chars:
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


def _is_continuation(goal: str, previous_topic: str) -> bool:
    if not previous_topic.strip():
        return False

    text = goal.strip().casefold()
    if re.search(
        r"^(?:その|それ|この|前回|先ほど|さっき|上記|上述|前の)|"
        r"^(?:that|those|these|previous)\b",
        text,
        flags=re.IGNORECASE,
    ):
        return True

    previous_tokens = {
        token
        for token in re.findall(r"[a-z0-9][a-z0-9_.:-]{2,}", previous_topic.casefold())
    }
    current_tokens = {
        token
        for token in re.findall(r"[a-z0-9][a-z0-9_.:-]{2,}", text)
    }
    return bool(previous_tokens and previous_tokens & current_tokens)


def _is_http_url(value: str) -> bool:
    try:
        parsed = urlparse(value)
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)
    except ValueError:
        return False


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = value.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result
