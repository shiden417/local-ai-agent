from __future__ import annotations

import json
from typing import Any


class ToolLoopGuard:
    """Detect repeated identical tool calls within one task."""

    def __init__(self, max_identical_calls: int = 2) -> None:
        if max_identical_calls < 2:
            raise ValueError("max_identical_calls must be at least 2")

        self.max_identical_calls = max_identical_calls
        self._counts: dict[tuple[str, str], int] = {}

    def record(self, name: str, arguments: dict[str, Any]) -> int:
        key = self._key(name, arguments)
        self._counts[key] = self._counts.get(key, 0) + 1
        return self._counts[key]

    def is_repetition(self, name: str, arguments: dict[str, Any]) -> bool:
        return (
            self._counts.get(self._key(name, arguments), 0)
            >= self.max_identical_calls
        )

    def reset(self) -> None:
        self._counts.clear()

    def reset_for_state_change(
        self,
        *,
        preserve_name: str,
        preserve_arguments: dict[str, Any],
    ) -> None:
        """Invalidate observations from the old state without replaying a mutation."""
        preserved_key = self._key(preserve_name, preserve_arguments)
        preserved_count = self._counts.get(preserved_key)
        self._counts.clear()
        if preserved_count is not None:
            # Keep the completed state-changing call blocked from replay.
            # The next observation is allowed because all other stale counts
            # were cleared above.
            self._counts[preserved_key] = self.max_identical_calls

    def message(self, name: str, arguments: dict[str, Any]) -> str:
        return (
            "このTaskでは同じTool呼び出しがすでに実行されています。"
            "同じ結果を再取得せず、既に得た観測結果を利用するか、"
            "別の方法で前進してください。"
            f" Tool={name}, arguments={json.dumps(arguments, ensure_ascii=False)}"
        )

    @staticmethod
    def _key(name: str, arguments: dict[str, Any]) -> tuple[str, str]:
        return (
            name,
            json.dumps(arguments, ensure_ascii=False, sort_keys=True),
        )
