from agent.loop_guard import ToolLoopGuard


def test_loop_guard_detects_repeated_call() -> None:
    guard = ToolLoopGuard(max_identical_calls=2)

    assert guard.record("list_directory", {"path": "."}) == 1
    assert guard.is_repetition("list_directory", {"path": "."}) is False

    guard.record("search_files", {"query": "x"})
    assert guard.is_repetition("list_directory", {"path": "."}) is False

    assert guard.record("list_directory", {"path": "."}) == 2
    assert guard.is_repetition("list_directory", {"path": "."}) is True


def test_loop_guard_distinguishes_arguments() -> None:
    guard = ToolLoopGuard(max_identical_calls=2)

    guard.record("read_file", {"path": "a.py"})

    assert guard.is_repetition("read_file", {"path": "b.py"}) is False


def test_loop_guard_can_reset() -> None:
    guard = ToolLoopGuard()
    guard.record("list_directory", {"path": "."})

    guard.reset()

    assert guard.is_repetition("list_directory", {"path": "."}) is False


def test_loop_guard_preserves_mutation_but_allows_new_state_observation() -> None:
    guard = ToolLoopGuard()
    guard.record("file_mutation", {"operation": "edit", "path": "test.txt"})
    guard.record("read_file", {"path": "test.txt"})

    guard.reset_for_state_change(
        preserve_name="file_mutation",
        preserve_arguments={"operation": "edit", "path": "test.txt"},
    )

    assert guard.is_repetition(
        "file_mutation",
        {"operation": "edit", "path": "test.txt"},
    )
    assert not guard.is_repetition("read_file", {"path": "test.txt"})
