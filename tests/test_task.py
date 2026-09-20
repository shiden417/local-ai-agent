from agent.task import TaskPhase, TaskState, TaskStatus


def test_task_lifecycle() -> None:
    task = TaskState(goal="完成作業")

    assert task.status == TaskStatus.PENDING
    assert task.phase == TaskPhase.UNDERSTAND

    task.start()
    task.begin_iteration()
    task.begin_iteration()
    task.record_tool("read_file", succeeded=True)
    assert task.phase == TaskPhase.VERIFY
    task.begin_iteration()
    assert task.phase == TaskPhase.ACT
    task.complete()

    assert task.status == TaskStatus.COMPLETED
    assert task.phase == TaskPhase.COMPLETE
    assert task.iteration == 3
    assert task.tool_calls == 1
    assert task.last_tool == "read_file"


def test_failed_tool_is_recorded() -> None:
    task = TaskState(goal="調査")
    task.start()
    task.record_tool("search_files", succeeded=False)

    assert task.phase == TaskPhase.VERIFY
    assert task.errors == ["Tool failed: search_files"]


def test_task_tracks_new_and_repeated_observations() -> None:
    task = TaskState(goal="調査")
    task.start()

    task.record_tool(
        "list_directory",
        succeeded=True,
        summary='{"entries":["a.py"]}',
        signature="same-result",
    )
    task.record_tool(
        "list_directory",
        succeeded=True,
        summary='{"entries":["a.py"]}',
        signature="same-result",
    )

    assert len(task.observations) == 2
    assert task.observations[0].new_information is True
    assert task.observations[1].new_information is False
    assert task.no_progress_streak == 1

    snapshot = task.snapshot()

    assert "Recent observations:" in snapshot
    assert "already_known" in snapshot
    assert "no_progress_streak=1" in snapshot


def test_max_iterations_state() -> None:
    task = TaskState(goal="長い作業")
    task.start()
    task.hit_max_iterations()

    assert task.status == TaskStatus.MAX_ITERATIONS
    assert task.phase == TaskPhase.UNDERSTAND
    assert "Maximum iterations reached." in task.errors[-1]


def test_snapshot_is_compact() -> None:
    task = TaskState(goal="確認")
    task.start()
    task.begin_iteration()

    snapshot = task.snapshot()

    assert "status=running" in snapshot
    assert "phase=plan" in snapshot
    assert "iteration=1" in snapshot
    assert "tool_calls=0" in snapshot
