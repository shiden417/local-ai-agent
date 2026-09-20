from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


DEFAULT_MAX_TURNS = 8
DEFAULT_MAX_CHARS = 8_000


@dataclass
class ConversationManager:
    """Keep short conversational context separate from task execution history."""

    max_turns: int = DEFAULT_MAX_TURNS
    max_chars: int = DEFAULT_MAX_CHARS
    _messages: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.max_turns < 1:
            raise ValueError("max_turns must be at least 1")
        if self.max_chars < 1:
            raise ValueError("max_chars must be positive")

    @property
    def messages(self) -> list[dict[str, Any]]:
        return [dict(message) for message in self._messages]

    def recent_messages(self) -> list[dict[str, Any]]:
        return [dict(message) for message in self._messages]

    def add_turn(self, user_content: str, assistant_content: str) -> None:
        user = user_content.strip()
        assistant = assistant_content.strip()
        if not user or not assistant:
            return

        self._messages.extend(
            [
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ]
        )
        self._trim()

    def clear(self) -> None:
        self._messages.clear()

    def _trim(self) -> None:
        while len(self._messages) > self.max_turns * 2:
            self._messages.pop(0)

        while (
            sum(len(str(message.get("content", ""))) + 40 for message in self._messages)
            > self.max_chars
            and len(self._messages) > 2
        ):
            self._messages.pop(0)
            self._messages.pop(0)
