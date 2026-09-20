from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_APPROVAL_PATH = Path.home() / ".local-ai-agent" / "approvals.json"


class ApprovalPolicy:
    """Persist explicit user approvals for repeatable local actions."""

    def __init__(self, path: str | Path = DEFAULT_APPROVAL_PATH) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
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

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
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
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps({"allowed": self._allowed}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(self.path)


def approval_key(tool_name: str, arguments: dict[str, Any], workspace: str | Path) -> str:
    """Return a stable, least-privilege key for a repeatable action."""
    tool = str(tool_name).strip()
    cwd = Path(workspace).resolve()

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


def _canonical_arguments(arguments: dict[str, Any]) -> str:
    return json.dumps(arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
