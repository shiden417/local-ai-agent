from __future__ import annotations

import json
from typing import Any


class ToolLoopGuard:
    """Detect consecutive identical tool calls that make no progress."""

    def __init__(self, max_identical_calls: int = 2) -> None:
        if max_identical_calls < 2:
            raise ValueError("max_identical_calls must be at least 2")

        self.max_identical_calls = max_identical_calls
        self._last_key: tuple[str, str] | None = None
        self._consecutive_count = 0

    def record(self, name: str, arguments: dict[str, Any]) -> int:
        key = self._key(name, arguments)

        if key == self._last_key:
            self._consecutive_count += 1
        else:
            self._last_key = key
            self._consecutive_count = 1

        return self._consecutive_count

    def is_repetition(self, name: str, arguments: dict[str, Any]) -> bool:
        return (
            self._key(name, arguments) == self._last_key
            and self._consecutive_count >= self.max_identical_calls
        )

    def reset(self) -> None:
        self._last_key = None
        self._consecutive_count = 0

    def message(self, name: str, arguments: dict[str, Any]) -> str:
        return (
            "このTool呼び出しは直前にも同じ引数で実行され、"
            "同じ結果を取得済みです。同じ操作を繰り返さず、"
            "既に得た結果を使って次の行動を選んでください。"
            f" Tool={name}, arguments={json.dumps(arguments, ensure_ascii=False)}"
        )

    @staticmethod
    def _key(name: str, arguments: dict[str, Any]) -> tuple[str, str]:
        return (
            name,
            json.dumps(arguments, ensure_ascii=False, sort_keys=True),
        )
