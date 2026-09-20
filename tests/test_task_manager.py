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



def test_task_manager_trims_old_completed_history() -> None:
    manager = TaskManager(max_tasks=2)

    first = manager.create("first")
    manager.complete(first.task_id)
    second = manager.create("second")
    manager.complete(second.task_id)
    third = manager.create("third")

    assert manager.get(first.task_id) is None
    assert manager.get(second.task_id) is not None
    assert manager.get(third.task_id) is not None


def test_task_manager_does_not_evict_active_current_task() -> None:
    manager = TaskManager(max_tasks=1)

    first = manager.create("active")
    second = manager.create("new")

    assert manager.get(first.task_id) is not None
    assert manager.get(second.task_id) is not None
