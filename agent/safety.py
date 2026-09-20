from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from agent.tool_registry import ToolRegistry


DEFAULT_APPROVAL_PATH = Path.home() / ".local-ai-agent" / "approvals.json"

AUTO_ALLOW = "allow"
AUTO_ASK = "ask"
AUTO_DENY = "deny"

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


class SafetyPolicy:
    """Single entry point for execution risk, scope checks, and learned approvals."""

    def __init__(self, approval_path: str | Path = DEFAULT_APPROVAL_PATH) -> None:
        self.approval_path = Path(approval_path).expanduser().resolve()
        self.approval_path.parent.mkdir(parents=True, exist_ok=True)
        self._allowed: dict[str, dict[str, str]] = {}
        self._load()

    def is_allowed(self, key: str) -> bool:
        return str(key).strip() in self._allowed

    def allow(self, key: str, description: str = "") -> None:
        key = str(key).strip()
        if not key:
            raise ValueError("approval key must not be empty")
        self._allowed[key] = {"description": str(description).strip()}
        self._save()

    def revoke(self, key: str) -> bool:
        key = str(key).strip()
        if key not in self._allowed:
            return False
        del self._allowed[key]
        self._save()
        return True

    def clear(self) -> None:
        self._allowed.clear()
        self._save()

    def entries(self) -> list[dict[str, str]]:
        return [
            {"key": key, **value}
            for key, value in sorted(self._allowed.items())
        ]

    def approval_key(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        workspace: str | Path,
    ) -> str:
        return approval_key(tool_name, arguments, workspace)

    def decide(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        registry: ToolRegistry,
        working_directory: str | Path | None = None,
    ) -> str:
        """Return allow, ask, or deny after applying all hard safety checks."""
        auto_decision = _classify_auto_mode(
            tool_name,
            arguments,
            registry,
            working_directory,
        )
        if auto_decision == AUTO_DENY:
            return AUTO_DENY

        if auto_decision == AUTO_ASK:
            key = self.approval_key(
                tool_name,
                arguments,
                working_directory or Path.cwd(),
            )
            if self.is_allowed(key):
                return AUTO_ALLOW
            return AUTO_ASK

        return AUTO_ALLOW

    def _load(self) -> None:
        if not self.approval_path.exists():
            return
        try:
            raw = json.loads(
                self.approval_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(raw, dict):
            return
        allowed = raw.get("allowed", {})
        if not isinstance(allowed, dict):
            return
        self._allowed = {
            str(key): value
            for key, value in allowed.items()
            if isinstance(value, dict)
        }

    def _save(self) -> None:
        temporary = self.approval_path.with_suffix(
            self.approval_path.suffix + ".tmp"
        )
        temporary.write_text(
            json.dumps(
                {"allowed": self._allowed},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        temporary.replace(self.approval_path)


def approval_key(
    tool_name: str,
    arguments: dict[str, Any],
    workspace: str | Path,
) -> str:
    """Return a stable, least-privilege key for a repeatable action."""
    tool = str(tool_name).strip()
    cwd = Path(workspace).resolve()
    _ = cwd

    if tool == "file_mutation":
        operation = str(arguments.get("operation", "")).strip().lower()
        path = str(arguments.get("path", "")).strip()
        if operation in {"create", "edit"}:
            return f"file_mutation:{operation}:workspace"
        if operation == "delete":
            return f"file_mutation:delete:{_digest(path)}"

    if tool == "run_python_script":
        return f"run_python_script:{_digest(str(arguments.get('script', '')))}"

    if tool == "execute_command":
        return f"execute_command:{_digest(str(arguments.get('command', '')).strip())}"

    if tool in {"stage_plugin", "promote_plugin", "test_plugin_candidate"}:
        return f"{tool}:{_digest(_canonical_arguments(arguments))}"

    return f"{tool}:{_digest(_canonical_arguments(arguments))}"


def _classify_auto_mode(
    tool_name: str,
    arguments: dict[str, Any],
    registry: ToolRegistry,
    working_directory: str | Path | None = None,
) -> str:
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

    if tool_name in {
        "list_directory",
        "read_file",
        "search_files",
        "list_promotion_candidates",
        "search_web",
    }:
        return AUTO_ALLOW

    if tool_name == "stage_plugin":
        return AUTO_ALLOW

    if tool_name in {
        "promote_plugin",
        "test_plugin_candidate",
        "run_python_script",
    }:
        return AUTO_ASK

    if tool is not None and not tool.requires_confirmation:
        return AUTO_ALLOW

    return AUTO_ASK



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

    match = re.match(r"[^\s'\" ]+", command[start:])
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


def _canonical_arguments(arguments: dict[str, Any]) -> str:
    return json.dumps(
        arguments,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


__all__ = [
    "AUTO_ALLOW",
    "AUTO_ASK",
    "AUTO_DENY",
    "DESTRUCTIVE_COMMAND_PATTERNS",
    "HARD_DENY_COMMAND_PATTERNS",
    "SafetyPolicy",
    "approval_key",
    "SafetyPolicy",
    "validate_command_scope",
]
