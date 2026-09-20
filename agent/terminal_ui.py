from __future__ import annotations

import os
import re
import shutil
import sys
from typing import Any


RESET = "\x1b[0m"
BOLD = "\x1b[1m"
CYAN = "\x1b[36m"
GREEN = "\x1b[32m"
YELLOW = "\x1b[33m"
RED = "\x1b[31m"
GRAY = "\x1b[90m"


class TerminalUI:
    """Small ANSI terminal UI with a plain-text fallback."""

    def __init__(self, model: str) -> None:
        self.model = model
        self.enabled = _supports_color()

    def _c(self, text: str, color: str) -> str:
        return f"{color}{text}{RESET}" if self.enabled else text

    def _line(self, label: str, value: str, width: int) -> str:
        content = f"│  {label:<10} {value}"
        padding = max(0, width - len(content) + 1)
        return self._c(content + (" " * padding), CYAN) + self._c("│", CYAN)
    def startup(self, workspace: str, approval_mode: str) -> None:
        width = max(54, min(shutil.get_terminal_size((80, 24)).columns, 88))
        inner = width - 4
        print()
        print(self._c("╭" + "─" * (width - 2) + "╮", CYAN))
        print(self._c("│", CYAN) + f"  {BOLD}Local AI Agent{RESET if self.enabled else ''}".ljust(inner) + self._c("│", CYAN))
        print(self._line("Model", self.model, inner))
        print(self._line("Workspace", workspace, inner))
        print(self._line("Approval", approval_mode, inner))
        print(self._c("╰" + "─" * (width - 2) + "╯", CYAN))
        print(self._c("Commands: /tasks  /permissions  /clear-permissions  /exit", GRAY))
        print()

    def task_start(self, goal: str, task_id: str) -> None:
        print(self._c(f"┌─ Task {task_id} ─────────────────────────────────", CYAN))
        print(f"│ {goal}")
        print(self._c("└────────────────────────────────────────────────", CYAN))

    def phase(self, phase: str, iteration: int) -> None:
        print(self._c(f"  [{phase.upper():8}] iteration {iteration}", YELLOW))

    def tool_start(self, name: str, arguments: dict[str, Any]) -> None:
        short = _compact(arguments)
        suffix = f"  {short}" if short else ""
        print(self._c(f"  ▶ {name}", CYAN) + suffix)

    def tool_result(self, ok: bool, summary: str) -> None:
        symbol = "✓" if ok else "✗"
        color = GREEN if ok else RED
        print(self._c(f"  {symbol} {summary}", color))

    def final(self, content: str) -> None:
        print(self._c("\n● Agent", GREEN))
        print(content)

    def error(self, content: str) -> None:
        print(self._c(f"✗ {content}", RED))

    def info(self, content: str) -> None:
        print(self._c(f"  {content}", GRAY))

    def approval(self, description: str) -> str:
        print(self._c("\n  Approval required", YELLOW))
        print(f"  {description}")
        return input("  [y] once / [a] always for this action / [n] deny: ").strip().lower()


def _compact(arguments: dict[str, Any], max_chars: int = 140) -> str:
    raw = repr(arguments)
    raw = re.sub(r"\s+", " ", raw)
    return raw if len(raw) <= max_chars else raw[: max_chars - 1] + "…"


def _supports_color() -> bool:
    if os.getenv("NO_COLOR") is not None:
        return False
    if "PYTEST_CURRENT_TEST" in os.environ:
        return False
    if sys.platform == "win32":
        return bool(os.getenv("WT_SESSION") or os.getenv("ANSICON")) or sys.stdout.isatty()
    return sys.stdout.isatty()
