from agent.task import classify_progress
from agent.task import ProgressState


def test_failed_result_is_not_progress_even_when_observation_is_new() -> None:
    evaluation = classify_progress(
        "read_file",
        {"ok": False, "error": "missing"},
        observation_is_new=True,
    )

    assert evaluation == ProgressState.FAILED


def test_repeated_result_is_blocked() -> None:
    evaluation = classify_progress(
        "read_file",
        {"ok": False, "repeated_tool_call": True},
        observation_is_new=False,
    )

    assert evaluation == ProgressState.BLOCKED


def test_new_successful_observation_is_progress() -> None:
    evaluation = classify_progress(
        "read_file",
        {"ok": True, "content": "new"},
        observation_is_new=True,
    )

    assert evaluation == ProgressState.PROGRESSED


def test_empty_memory_search_is_not_progress() -> None:
    evaluation = classify_progress(
        "search_memory",
        {"ok": True, "entries": []},
        observation_is_new=True,
    )

    assert evaluation == ProgressState.NO_PROGRESS
