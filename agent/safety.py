from __future__ import annotations

import re
from typing import Any


DESTRUCTIVE_COMMAND_PATTERNS = (
    re.compile(r"\bremove-item\b", re.IGNORECASE),
    re.compile(r"\bdel(?:ete)?\b", re.IGNORECASE),
    re.compile(r"\berase\b", re.IGNORECASE),
    re.compile(r"\brmdir\b", re.IGNORECASE),
    re.compile(r"\bformat(?:-volume)?\b", re.IGNORECASE),
    re.compile(r"\bstop-process\b", re.IGNORECASE),
    re.compile(r"\bgit\s+reset\b.*--hard\b", re.IGNORECASE),
    re.compile(r"\bgit\s+clean\b.*(?:-f|--force)\b", re.IGNORECASE),
    re.compile(r"\bgit\s+checkout\s+--\b", re.IGNORECASE),
    re.compile(r"\bgit\s+restore\b.*(?:--source|-s)\b", re.IGNORECASE),
)


def requires_confirmation(
    tool_name: str,
    arguments: dict[str, Any],
) -> bool:
    if tool_name == "edit_file":
        return True

    if tool_name != "execute_command":
        return False

    command = str(arguments.get("command", ""))
    return any(pattern.search(command) for pattern in DESTRUCTIVE_COMMAND_PATTERNS)
