from __future__ import annotations

import json
from collections import deque
from typing import Any


class ToolLoopGuard:
    """Detect repeated tool calls that make no progress."""

    def __init__(self, max_identical_calls: int = 2, history_size: int = 20) -> None:
        if max_identical_calls < 1:
            raise ValueError("max_identical_calls must be positive")
        if history_size < 1:
            raise ValueError("history_size must be positive")

        self.max_identical_calls = max_identical_calls
        self._recent_calls: deque[tuple[str, str]] = deque(maxlen=history_size)
        self._counts: dict[tuple[str, str], int] = {}

    def record(self, name: str, arguments: dict[str, Any]) -> int:
        key = (
            name,
            json.dumps(arguments, ensure_ascii=False, sort_keys=True),
        )
        self._recent_calls.append(key)
        self._counts[key] = self._counts.get(key, 0) + 1
        return self._counts[key]

    def is_repetition(self, name: str, arguments: dict[str, Any]) -> bool:
        key = (
            name,
            json.dumps(arguments, ensure_ascii=False, sort_keys=True),
        )
        return self._counts.get(key, 0) >= self.max_identical_calls

    def reset(self) -> None:
        self._recent_calls.clear()
        self._counts.clear()

    def message(self, name: str, arguments: dict[str, Any]) -> str:
        return (
            "このTool呼び出しは直前にも実行され、同じ結果を取得済みです。"
            "同じToolを繰り返さず、既に得た結果を使って次の行動を選んでください。"
            f" Tool={name}, arguments={json.dumps(arguments, ensure_ascii=False)}"
        )
