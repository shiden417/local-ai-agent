from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4

from agent.task import TaskState, TaskStatus


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ManagedTask:
    """A task record owned by the Agent TaskManager."""

    task_id: str
    state: TaskState
    created_order: int
    messages: list[dict] = field(default_factory=list)
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)

    @property
    def goal(self) -> str:
        return self.state.goal

    @property
    def status(self) -> TaskStatus:
        return self.state.status


class TaskManager:
    """Manage multiple independent Agent tasks in memory."""

    def __init__(self) -> None:
        self._tasks: dict[str, ManagedTask] = {}
        self._current_task_id: str | None = None
        self._next_order = 0

    def create(self, goal: str) -> ManagedTask:
        goal = goal.strip()
        if not goal:
            raise ValueError("task goal must not be empty")

        self._next_order += 1
        task = ManagedTask(
            task_id=uuid4().hex[:12],
            state=TaskState(goal=goal),
            created_order=self._next_order,
        )
        self._tasks[task.task_id] = task
        self._current_task_id = task.task_id
        return task

    def get(self, task_id: str) -> ManagedTask | None:
        return self._tasks.get(task_id)

    @property
    def current(self) -> ManagedTask | None:
        if self._current_task_id is None:
            return None
        return self.get(self._current_task_id)

    def select(self, task_id: str) -> ManagedTask:
        task = self.get(task_id)
        if task is None:
            raise KeyError(f"Unknown task: {task_id}")
        self._current_task_id = task_id
        return task

    def list_tasks(self) -> list[ManagedTask]:
        return sorted(
            self._tasks.values(),
            key=lambda task: task.created_order,
            reverse=True,
        )

    def active_tasks(self) -> list[ManagedTask]:
        return [
            task
            for task in self.list_tasks()
            if task.status in {TaskStatus.PENDING, TaskStatus.RUNNING}
        ]

    def update_timestamp(self, task: ManagedTask) -> None:
        task.updated_at = _now()

    def complete(self, task_id: str) -> ManagedTask:
        task = self._require(task_id)
        task.state.complete()
        self.update_timestamp(task)
        return task

    def fail(self, task_id: str, error: str) -> ManagedTask:
        task = self._require(task_id)
        task.state.fail(error)
        self.update_timestamp(task)
        return task

    def _require(self, task_id: str) -> ManagedTask:
        task = self.get(task_id)
        if task is None:
            raise KeyError(f"Unknown task: {task_id}")
        return task
