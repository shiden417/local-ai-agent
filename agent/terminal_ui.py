from __future__ import annotations

import os
import re
import shutil
import sys
import threading
import time
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
        self._spinner_stop: threading.Event | None = None
        self._spinner_thread: threading.Thread | None = None
        self._spinner_lock = threading.Lock()
        self._activity_label = ""

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
        title = f"  {BOLD}J.A.R.V.I.S. Local Agent{RESET if self.enabled else ''}"
        print(self._c("│", CYAN) + title.ljust(inner) + self._c("│", CYAN))
        print(self._line("Model", self.model, inner))
        print(self._line("Workspace", workspace, inner))
        print(self._line("Approval", approval_mode, inner))
        print(self._c("╰" + "─" * (width - 2) + "╯", CYAN))
        print(self._c("Commands: /tasks  /permissions  /clear-permissions  /clear-context  /exit", GRAY))
        print()

    def task_start(self, goal: str, task_id: str) -> None:
        print(self._c(f"  Task {task_id}: {goal}", CYAN))

    def phase(self, phase: str, iteration: int) -> None:
        # Phase/iteration details are useful internally but too noisy for the
        # normal interactive UI. The spinner communicates that work is ongoing.
        return

    def thinking_start(self, label: str = "Thinking") -> None:
        self.start_activity(label)

    def thinking_stop(self) -> None:
        self.stop_activity()

    def tool_start(self, name: str, arguments: dict[str, Any]) -> None:
        self.start_activity(f"Running {name}")

    def tool_result(self, ok: bool, summary: str) -> None:
        label = self._activity_label or "Tool"
        self.stop_activity()
        symbol = "✓" if ok else "✗"
        color = GREEN if ok else RED
        if ok:
            print(self._c(f"  {symbol} {label.removeprefix('Running ')}", color))
        else:
            compact_summary = " ".join(str(summary).split())
            if len(compact_summary) > 180:
                compact_summary = compact_summary[:177] + "..."
            print(self._c(f"  {symbol} {label.removeprefix('Running ')}: {compact_summary}", color))

    def start_activity(self, label: str) -> None:
        self.stop_activity()
        self._activity_label = label
        if not self.enabled:
            return

        stop_event = threading.Event()
        self._spinner_stop = stop_event
        self._spinner_thread = threading.Thread(
            target=self._run_spinner,
            args=(stop_event, label),
            name="terminal-spinner",
            daemon=True,
        )
        self._spinner_thread.start()

    def stop_activity(self) -> None:
        with self._spinner_lock:
            stop_event = self._spinner_stop
            thread = self._spinner_thread
            self._spinner_stop = None
            self._spinner_thread = None

        if stop_event is None:
            return

        stop_event.set()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=0.5)

        self._activity_label = ""
        if self.enabled:
            sys.stdout.write("\r\x1b[2K")
            sys.stdout.flush()

    def _run_spinner(
        self,
        stop_event: threading.Event,
        label: str,
    ) -> None:
        frames = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
        index = 0
        while not stop_event.is_set():
            frame = frames[index % len(frames)]
            sys.stdout.write(f"\r\x1b[2K  {frame} {label}")
            sys.stdout.flush()
            index += 1
            stop_event.wait(0.08)

    def final(self, content: str) -> None:
        print(self._c("\n● Agent", GREEN))
        print(content)

    def error(self, content: str) -> None:
        self.stop_activity()
        print(self._c(f"✗ {content}", RED))

    def info(self, content: str) -> None:
        self.stop_activity()
        print(self._c(f"  {content}", GRAY))

    def approval(self, description: str) -> str:
        self.stop_activity()
        print(self._c("\n  Approval required", YELLOW))
        print(f"  {description}")
        return input("  [y] once / [a] always for this action / [n] deny: ").strip().lower()

    def question(self, question: str) -> str:
        self.stop_activity()
        print(self._c("\n  Agent question", YELLOW))
        print(f"  {question}")
        return input("  Answer: ").strip()


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
