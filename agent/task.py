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
class TaskObservation:
    """Compact record of one tool observation."""

    tool: str
    ok: bool
    summary: str
    signature: str
    new_information: bool


@dataclass
class TaskState:
    """Generic execution state and observation ledger for one Agent task."""

    goal: str
    status: TaskStatus = TaskStatus.PENDING
    phase: TaskPhase = TaskPhase.UNDERSTAND
    iteration: int = 0
    tool_calls: int = 0
    last_tool: str | None = None
    errors: list[str] = field(default_factory=list)
    observations: list[TaskObservation] = field(default_factory=list)
    observation_signatures: set[str] = field(
        default_factory=set,
        repr=False,
    )
    no_progress_streak: int = 0

    def start(self) -> None:
        self.status = TaskStatus.RUNNING
        self.phase = TaskPhase.UNDERSTAND

    def begin_iteration(self) -> None:
        self.iteration += 1
        if self.phase == TaskPhase.UNDERSTAND:
            self.phase = TaskPhase.PLAN
        elif self.phase in {TaskPhase.PLAN, TaskPhase.VERIFY}:
            self.phase = TaskPhase.ACT

    def record_tool(
        self,
        name: str,
        succeeded: bool,
        summary: str = "",
        signature: str = "",
    ) -> None:
        self.tool_calls += 1
        self.last_tool = name
        self.phase = TaskPhase.VERIFY

        new_information = (
            not signature or signature not in self.observation_signatures
        )
        if signature:
            if new_information:
                self.observation_signatures.add(signature)
                self.no_progress_streak = 0
            else:
                self.no_progress_streak += 1
        else:
            self.no_progress_streak = 0

        self.observations.append(
            TaskObservation(
                tool=name,
                ok=succeeded,
                summary=summary,
                signature=signature,
                new_information=new_information,
            )
        )
        if len(self.observations) > 8:
            self.observations.pop(0)

        if not succeeded:
            self.errors.append(f"Tool failed: {name}")
            if len(self.errors) > 8:
                self.errors.pop(0)

    def complete(self) -> None:
        self.status = TaskStatus.COMPLETED
        self.phase = TaskPhase.COMPLETE

    def fail(self, error: str) -> None:
        self.status = TaskStatus.FAILED
        self.errors.append(error)
        if len(self.errors) > 8:
            self.errors.pop(0)

    def hit_max_iterations(self) -> None:
        self.status = TaskStatus.MAX_ITERATIONS
        self.errors.append("Maximum iterations reached.")

    def snapshot(self) -> str:
        last_tool = self.last_tool or "none"
        lines = [
            f"Task status={self.status.value}; "
            f"phase={self.phase.value}; "
            f"iteration={self.iteration}; "
            f"tool_calls={self.tool_calls}; "
            f"last_tool={last_tool}; "
            f"no_progress_streak={self.no_progress_streak}"
        ]

        if self.observations:
            lines.append("Recent observations:")
            for observation in self.observations[-6:]:
                status = "ok" if observation.ok else "failed"
                novelty = "new" if observation.new_information else "already_known"
                lines.append(
                    f"- {observation.tool}: {status}, {novelty}; "
                    f"{observation.summary}"
                )
        else:
            lines.append("Recent observations: none")

        return "\n".join(lines)
