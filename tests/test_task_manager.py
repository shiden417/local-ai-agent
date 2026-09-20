from agent.task import TaskStatus
from agent.task_manager import TaskManager


def test_task_manager_creates_and_selects_tasks() -> None:
    manager = TaskManager()

    first = manager.create("first task")
    second = manager.create("second task")

    assert first.task_id != second.task_id
    assert manager.current is second

    selected = manager.select(first.task_id)

    assert selected.task_id == first.task_id
    assert manager.current is first


def test_task_manager_keeps_tasks_independent() -> None:
    manager = TaskManager()

    first = manager.create("first")
    second = manager.create("second")

    first.state.start()
    first.state.begin_iteration()
    manager.complete(first.task_id)

    second.state.start()

    assert first.status == TaskStatus.COMPLETED
    assert second.status == TaskStatus.RUNNING
    assert manager.active_tasks() == [second]


def test_task_manager_lists_newest_first() -> None:
    manager = TaskManager()

    first = manager.create("first")
    second = manager.create("second")

    tasks = manager.list_tasks()

    assert [task.task_id for task in tasks] == [second.task_id, first.task_id]


def test_task_manager_rejects_unknown_task() -> None:
    manager = TaskManager()

    try:
        manager.select("missing")
    except KeyError as exc:
        assert "Unknown task" in str(exc)
    else:
        raise AssertionError("Expected unknown task error")
