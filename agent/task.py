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


class ProgressState(str, Enum):
    UNKNOWN = "unknown"
    PROGRESSED = "progressed"
    NO_PROGRESS = "no_progress"
    BLOCKED = "blocked"
    FAILED = "failed"


@dataclass
class TaskObservation:
    """Compact record of one tool observation."""

    tool: str
    ok: bool
    summary: str
    signature: str
    new_information: bool
    progress_state: ProgressState


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
    disabled_tools: set[str] = field(default_factory=set, repr=False)
    no_progress_streak: int = 0
    progress_state: ProgressState = ProgressState.UNKNOWN
    progress_count: int = 0
    recovery_tool: str | None = None

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
        new_information: bool | None = None,
        progress_state: ProgressState | None = None,
    ) -> None:
        self.tool_calls += 1
        self.last_tool = name
        self.phase = TaskPhase.VERIFY

        if new_information is None:
            new_information = (
                bool(signature)
                and signature not in self.observation_signatures
            )

        if signature and new_information:
            self.observation_signatures.add(signature)

        if progress_state is None:
            progress_state = (
                ProgressState.PROGRESSED
                if succeeded and new_information
                else ProgressState.FAILED
                if not succeeded
                else ProgressState.NO_PROGRESS
            )

        self.progress_state = progress_state

        if not succeeded:
            self.recovery_tool = name
        elif self.recovery_tool == name:
            # Keep the failed tool quarantined until a different successful
            # observation gives the model new evidence.
            self.recovery_tool = None
        elif progress_state == ProgressState.PROGRESSED:
            # A useful alternative observation completes the recovery step.
            self.recovery_tool = None

        if progress_state == ProgressState.PROGRESSED:
            self.progress_count += 1
            self.no_progress_streak = 0
        else:
            self.no_progress_streak += 1

        self.observations.append(
            TaskObservation(
                tool=name,
                ok=succeeded,
                summary=summary,
                signature=signature,
                new_information=new_information,
                progress_state=progress_state,
            )
        )
        if len(self.observations) > 8:
            self.observations.pop(0)

        if not succeeded:
            self.errors.append(f"Tool failed: {name}")
            if len(self.errors) > 8:
                self.errors.pop(0)

    def disable_tool(self, name: str) -> None:
        self.disabled_tools.add(name)

    def is_tool_disabled(self, name: str) -> bool:
        return name in self.disabled_tools

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
        disabled = ", ".join(sorted(self.disabled_tools)) or "none"
        recovery = self.recovery_tool or "none"
        lines = [
            (
                f"Task status={self.status.value}; "
                f"phase={self.phase.value}; "
                f"iteration={self.iteration}; "
                f"tool_calls={self.tool_calls}; "
                f"last_tool={last_tool}; "
                f"progress_state={self.progress_state.value}; "
                f"progress_count={self.progress_count}; "
                f"no_progress_streak={self.no_progress_streak}"
            ),
            f"Disabled tools: {disabled}",
            f"Recovery quarantine: {recovery}",
            "Execution guidance: "
            "use the smallest action that advances the goal; "
            "after a useful observation, verify whether the goal can already be answered; "
            "do not repeat known observations.",
        ]

        if self.phase == TaskPhase.PLAN:
            lines.append(
                "Current focus: choose the first concrete action needed for the goal."
            )
        elif self.phase == TaskPhase.ACT:
            lines.append(
                "Current focus: execute one concrete action that is not already known."
            )
        elif self.phase == TaskPhase.VERIFY:
            lines.append(
                "Current focus: verify the latest observation, then answer or choose the next action."
            )

        if self.observations:
            lines.append("Recent observations:")
            for observation in self.observations[-6:]:
                status = "ok" if observation.ok else "failed"
                novelty = (
                    "new" if observation.new_information else "already_known"
                )
                lines.append(
                    f"- {observation.tool}: {status}, {novelty}, "
                    f"{observation.progress_state.value}; {observation.summary}"
                )
        else:
            lines.append("Recent observations: none")

        return "\n".join(lines)
