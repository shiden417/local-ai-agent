from __future__ import annotations

import re
from typing import Any

from agent.tool_registry import ToolRegistry


# This is a conservative heuristic, not a complete security policy.
DESTRUCTIVE_COMMAND_PATTERNS = (
    re.compile(r"\bremove-item\b", re.IGNORECASE),
    re.compile(r"\bset-content\b", re.IGNORECASE),
    re.compile(r"\badd-content\b", re.IGNORECASE),
    re.compile(r"\bclear-content\b", re.IGNORECASE),
    re.compile(r"\bout-file\b", re.IGNORECASE),
    re.compile(r"\bnew-item\b", re.IGNORECASE),
    re.compile(r"\bcopy-item\b", re.IGNORECASE),
    re.compile(r"\bmove-item\b", re.IGNORECASE),
    re.compile(r"\brename-item\b", re.IGNORECASE),
    re.compile(r"\bdel(?:ete)?\b", re.IGNORECASE),
    re.compile(r"\berase\b", re.IGNORECASE),
    re.compile(r"\brmdir\b", re.IGNORECASE),
    re.compile(r"\bformat(?:-volume)?\b", re.IGNORECASE),
    re.compile(r"\bstop-process\b", re.IGNORECASE),
    re.compile(r"\bgit\s+reset\b.*--hard\b", re.IGNORECASE),
    re.compile(r"\bgit\s+clean\b.*(?:-f|--force)\b", re.IGNORECASE),
    re.compile(r"\bgit\s+checkout\s+--\b", re.IGNORECASE),
    re.compile(r"\bgit\s+restore\b.*(?:--source|-s)\b", re.IGNORECASE),
    re.compile(r"(^|[^-])>>?", re.IGNORECASE),
)


def requires_confirmation(
    tool_name: str,
    arguments: dict[str, Any],
    registry: ToolRegistry,
) -> bool:
    tool = registry.get(tool_name)
    if tool is not None and tool.requires_confirmation:
        return True

    if tool_name != "execute_command":
        return False

    command = str(arguments.get("command", ""))
    return any(pattern.search(command) for pattern in DESTRUCTIVE_COMMAND_PATTERNS)
