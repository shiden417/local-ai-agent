from __future__ import annotations

import re
from pathlib import Path
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


WINDOWS_ABSOLUTE_PATH_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])(?:[A-Za-z]:\\|\\\\|/(?:mnt|var|etc|tmp)(?:/|$))"
)
PARENT_PATH_PATTERN = re.compile(r"(^|[\s'\"])(?:\.\.[\\/])+")
LOCAL_PATH_MUTATING_TOOLS = {"file_mutation"}

AUTO_ALLOW = "allow"
AUTO_ASK = "ask"
AUTO_DENY = "deny"

HARD_DENY_COMMAND_PATTERNS = (
    re.compile(r"\bshutdown(?:\.exe)?\b", re.IGNORECASE),
    re.compile(r"\brestart-computer\b", re.IGNORECASE),
    re.compile(r"\bstop-computer\b", re.IGNORECASE),
    re.compile(r"\bclear-disk\b", re.IGNORECASE),
    re.compile(r"\bformat-volume\b", re.IGNORECASE),
    re.compile(r"\bdiskpart\b", re.IGNORECASE),
    re.compile(r"\bgit\s+push\b.*(?:--force|-f)\b", re.IGNORECASE),
)
NETWORK_COMMAND_PATTERNS = (
    re.compile(r"\bcurl(?:\.exe)?\b", re.IGNORECASE),
    re.compile(r"\binvoke-webrequest\b", re.IGNORECASE),
    re.compile(r"\binvoke-restmethod\b", re.IGNORECASE),
    re.compile(r"\bwget(?:\.exe)?\b", re.IGNORECASE),
    re.compile(r"\b(?:pip|python)\s+.*\binstall\b", re.IGNORECASE),
)


def classify_auto_mode(
    tool_name: str,
    arguments: dict[str, Any],
    registry: ToolRegistry,
    working_directory: str | Path | None = None,
) -> str:
    """Classify an action for the default always-on Auto Mode."""
    tool = registry.get(tool_name)

    if tool_name == "execute_command":
        command = str(arguments.get("command", ""))
        if any(pattern.search(command) for pattern in HARD_DENY_COMMAND_PATTERNS):
            return AUTO_DENY
        if working_directory is not None:
            scope_error = validate_command_scope(command, working_directory)
            if scope_error is not None:
                return AUTO_DENY
        if any(pattern.search(command) for pattern in DESTRUCTIVE_COMMAND_PATTERNS):
            return AUTO_ASK
        if any(pattern.search(command) for pattern in NETWORK_COMMAND_PATTERNS):
            return AUTO_ASK
        return AUTO_ALLOW

    if tool_name == "file_mutation":
        operation = str(arguments.get("operation", "")).strip().lower()
        if operation in {"create", "edit"}:
            requested_path = str(arguments.get("path", "")).strip()
            if working_directory is not None and requested_path:
                try:
                    target = Path(requested_path).expanduser().resolve()
                    cwd = Path(working_directory).resolve()
                    if target.is_relative_to(cwd):
                        return AUTO_ALLOW
                except OSError:
                    pass
        return AUTO_ASK

    if tool_name in {"list_directory", "read_file", "search_files", "list_promotion_candidates", "search_web"}:
        return AUTO_ALLOW

    if tool_name == "stage_plugin":
        return AUTO_ALLOW

    if tool_name in {"promote_plugin", "test_plugin_candidate", "run_python_script"}:
        return AUTO_ASK

    if tool is not None and not tool.requires_confirmation:
        return AUTO_ALLOW

    return AUTO_ASK

def requires_confirmation(
    tool_name: str,
    arguments: dict[str, Any],
    registry: ToolRegistry,
    working_directory: str | Path | None = None,
) -> bool:
    tool = registry.get(tool_name)
    if tool is not None and tool.requires_confirmation:
        return True

    if tool_name in LOCAL_PATH_MUTATING_TOOLS:
        requested_path = str(arguments.get("path", "")).strip()
        if _is_absolute_local_path(requested_path):
            if working_directory is None:
                return True
            try:
                target = Path(requested_path).resolve()
                cwd = Path(working_directory).resolve()
            except OSError:
                return True
            if not target.is_relative_to(cwd):
                return True

    if tool_name != "execute_command":
        return False

    command = str(arguments.get("command", ""))
    if any(pattern.search(command) for pattern in DESTRUCTIVE_COMMAND_PATTERNS):
        return True

    # Commands referencing absolute paths or traversing above the current
    # workspace require explicit user confirmation. Runtime separately
    # rejects clear workspace escapes before execution.
    return bool(
        WINDOWS_ABSOLUTE_PATH_PATTERN.search(command)
        or PARENT_PATH_PATTERN.search(command)
    )


def validate_command_scope(
    command: str,
    working_directory: str | Path,
) -> str | None:
    """Reject obvious PowerShell path escapes from the Agent workspace."""
    cwd = Path(working_directory).resolve()
    text = str(command)

    for match in WINDOWS_ABSOLUTE_PATH_PATTERN.finditer(text):
        token = _extract_path_token(text, match.start())
        if not token:
            return (
                "PowerShell command contains an absolute path. "
                "Absolute paths are not allowed by the current workspace policy."
            )
        try:
            candidate = Path(token).expanduser().resolve()
        except OSError:
            return (
                "PowerShell command contains a path that could not be resolved."
            )
        if not candidate.is_relative_to(cwd):
            return (
                "PowerShell command targets a path outside the Agent workspace: "
                f"{candidate}"
            )

    if PARENT_PATH_PATTERN.search(text):
        return (
            "PowerShell command contains parent-directory traversal. "
            "Use paths that remain inside the Agent workspace."
        )

    return None


def _extract_path_token(command: str, start: int) -> str | None:
    """Extract a path while preserving spaces inside PowerShell quotes."""
    if start < 0 or start >= len(command):
        return None

    quote = None
    if start > 0 and command[start - 1] in {"'", "\""}:
        quote = command[start - 1]

    if quote is not None:
        end = command.find(quote, start)
        if end == -1:
            return None
        return command[start:end]

    match = re.match(r"[^\s'\"]+", command[start:])
    if not match:
        return None

    return match.group(0)


def _is_absolute_local_path(value: str) -> bool:
    if not value:
        return False
    return bool(
        Path(value).is_absolute()
        or re.match(r"^(?:[A-Za-z]:[\\/]|\\\\)", value)
    )
