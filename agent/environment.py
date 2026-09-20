from __future__ import annotations

import os
import platform
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Iterable

from agent.observation import truncate_text


MAX_TOP_LEVEL_ENTRIES = 24
MAX_AGENTS_FILES = 8
MAX_AGENTS_TOTAL_CHARS = 8_000
WINDOWS_ABSOLUTE_PATH = re.compile(r"[A-Za-z]:[\\/][^\s\"'<>|]+")
RELATIVE_PATH = re.compile(
    r"(?<![A-Za-z0-9_])(?:[A-Za-z0-9_.-]+[\\/])+(?:[A-Za-z0-9_.-]+)"
    r"(?:\.[A-Za-z0-9_.-]+)?"
)


def build_environment_context(
    workspace: str | Path,
    related_paths: Iterable[str] = (),
) -> str:
    root = Path(workspace).resolve()
    lines = [
        "[Environment]",
        f"OS: {platform.system()} {platform.release()}",
        f"Workspace: {root}",
        f"Time: {datetime.now().astimezone().isoformat(timespec='seconds')}",
    ]

    branch, status = _git_state(root)
    if branch:
        lines.append(f"Git: branch={branch}")
    else:
        lines.append("Git: unavailable")
    if branch:
        lines.append(f"Git status: {status or 'clean'}")

    entries = _top_level_entries(root)
    if entries:
        lines.append("Workspace root entries: " + ", ".join(entries))
        lines.append("Use list_directory for more detail; do not assume unseen paths.")

    agents = load_relevant_agents(root, related_paths)
    if agents:
        lines.append("")
        lines.append("[AGENTS.md Rules]")
        lines.append(
            "Rules are ordered from workspace-wide to more local directories. "
            "When rules conflict, the more local AGENTS.md takes precedence. "
            "Relative paths in a rule are resolved from that AGENTS.md directory."
        )
        lines.extend(agents)

    return "\n".join(lines)


def load_relevant_agents(
    workspace: Path,
    related_paths: Iterable[str] = (),
) -> list[str]:
    root = workspace.resolve()
    agent_paths: list[Path] = []

    root_agents = root / "AGENTS.md"
    if root_agents.is_file():
        agent_paths.append(root_agents)

    for raw_path in related_paths:
        candidate = _resolve_related_path(root, raw_path)
        if candidate is None:
            continue
        directory = candidate if candidate.is_dir() else candidate.parent
        if not directory.is_relative_to(root):
            continue

        current = directory.resolve()
        stack: list[Path] = []
        while True:
            agent_file = current / "AGENTS.md"
            if agent_file.is_file():
                stack.append(agent_file)
            if current == root:
                break
            current = current.parent
        for item in reversed(stack):
            if item not in agent_paths:
                agent_paths.append(item)
        if len(agent_paths) >= MAX_AGENTS_FILES:
            break

    blocks: list[str] = []
    total = 0
    for path in agent_paths[:MAX_AGENTS_FILES]:
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = path.read_text(encoding="cp932", errors="replace")
        except OSError:
            continue

        bounded, _ = truncate_text(content.strip(), max(500, MAX_AGENTS_TOTAL_CHARS - total))
        if not bounded:
            continue

        try:
            display = path.relative_to(root).as_posix()
        except ValueError:
            display = str(path)
        block = f"--- {display} ---\n{bounded}"
        blocks.append(block)
        total += len(block)
        if total >= MAX_AGENTS_TOTAL_CHARS:
            break

    return blocks


def extract_related_paths(text: str) -> list[str]:
    candidates: list[str] = []
    for pattern in (WINDOWS_ABSOLUTE_PATH, RELATIVE_PATH):
        for match in pattern.finditer(text):
            value = match.group(0).rstrip(".,;:)]}、。")
            if value not in candidates:
                candidates.append(value)
    return candidates[:12]


def _resolve_related_path(root: Path, raw_path: str) -> Path | None:
    value = str(raw_path).strip()
    value = value.strip('"').strip("'")
    if not value:
        return None
    try:
        candidate = Path(value).expanduser()
        if candidate.is_absolute():
            return candidate.resolve()
        candidate = (root / candidate).resolve()
        if candidate.is_relative_to(root):
            return candidate
    except (OSError, ValueError):
        return None
    return None


def _top_level_entries(root: Path) -> list[str]:
    try:
        entries = sorted(
            root.iterdir(),
            key=lambda item: (not item.is_dir(), item.name.lower()),
        )
    except OSError:
        return []

    names: list[str] = []
    for entry in entries[:MAX_TOP_LEVEL_ENTRIES]:
        names.append(entry.name + "/" if entry.is_dir() else entry.name)
    return names


def _git_state(root: Path) -> tuple[str | None, str]:
    try:
        branch_result = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=2,
            check=False,
        )
        if branch_result.returncode != 0:
            return None, ""
        branch = branch_result.stdout.strip()
        status_result = subprocess.run(
            ["git", "status", "--short"],
            cwd=root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=2,
            check=False,
        )
        status = "dirty" if status_result.stdout.strip() else "clean"
        return branch or None, status
    except (OSError, subprocess.SubprocessError):
        return None, ""
