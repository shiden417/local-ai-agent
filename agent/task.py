from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    MAX_ITERATIONS = "max_iterations"


class TaskPhase(str, Enum):
    UNDERSTAND = "understand"
    PLAN = "plan"
    ACT = "act"
    VERIFY = "verify"
    COMPLETE = "complete"


@dataclass
class TaskState:
    """Small, generic state container for one Agent task."""

    goal: str
    status: TaskStatus = TaskStatus.PENDING
    phase: TaskPhase = TaskPhase.UNDERSTAND
    iteration: int = 0
    tool_calls: int = 0
    last_tool: str | None = None
    errors: list[str] = field(default_factory=list)

    def start(self) -> None:
        self.status = TaskStatus.RUNNING
        self.phase = TaskPhase.UNDERSTAND

    def begin_iteration(self) -> None:
        self.iteration += 1
        if self.phase == TaskPhase.UNDERSTAND:
            self.phase = TaskPhase.PLAN
        elif self.phase in {TaskPhase.PLAN, TaskPhase.VERIFY}:
            self.phase = TaskPhase.ACT

    def record_tool(self, name: str, succeeded: bool) -> None:
        self.tool_calls += 1
        self.last_tool = name
        self.phase = TaskPhase.VERIFY
        if not succeeded:
            self.errors.append(f"Tool failed: {name}")

    def complete(self) -> None:
        self.status = TaskStatus.COMPLETED
        self.phase = TaskPhase.COMPLETE

    def fail(self, error: str) -> None:
        self.status = TaskStatus.FAILED
        self.errors.append(error)

    def hit_max_iterations(self) -> None:
        self.status = TaskStatus.MAX_ITERATIONS
        self.errors.append("Maximum iterations reached.")

    def snapshot(self) -> str:
        last_tool = self.last_tool or "none"
        return (
            f"Task status={self.status.value}; "
            f"phase={self.phase.value}; "
            f"iteration={self.iteration}; "
            f"tool_calls={self.tool_calls}; "
            f"last_tool={last_tool}"
        )
