from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class RequestMode(str, Enum):
    DIRECT = "direct"
    TASK = "task"


@dataclass(frozen=True)
class RequestClassification:
    mode: RequestMode


class RequestClassifier:
    """Identify only whether input is ordinary conversation or an Agent task.

    Tool selection itself is intentionally delegated to the model's Tool
    Calling. The classifier does not infer capabilities or concrete tools.
    """

    _DIRECT_PATTERNS: tuple[str, ...] = (
        r"^\s*(こんにちは|こんばんは|おはよう|やあ)[！!。\s]*$",
        r"^\s*(ありがとう|どうもありがとう|thanks|thank you)[！!。\s]*$",
        r"^\s*(さようなら|またね|bye)[！!。\s]*$",
        r"^\s*(元気ですか|元気？|元気\?)\s*$",
    )

    def classify(self, request_text: str) -> RequestClassification:
        text = request_text.strip()
        if any(
            re.search(pattern, text, flags=re.IGNORECASE)
            for pattern in self._DIRECT_PATTERNS
        ):
            return RequestClassification(RequestMode.DIRECT)
        return RequestClassification(RequestMode.TASK)
