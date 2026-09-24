from agent.recovery import (
    STATUS_BLOCKED,
    STATUS_EXTERNAL_FAILURE,
    STATUS_INVALID_INPUT,
    STATUS_NOT_FOUND,
    STATUS_SUCCESS,
    STATUS_TIMEOUT,
    classify_tool_outcome,
    recovery_guidance,
)


def test_classify_not_found() -> None:
    assert classify_tool_outcome(
        "read_file",
        {"ok": False, "error": "File does not exist: docs/missing.md"},
    ) == STATUS_NOT_FOUND


def test_classify_timeout() -> None:
    assert classify_tool_outcome(
        "execute_command",
        {"ok": False, "timed_out": True, "stderr": "timed out"},
    ) == STATUS_TIMEOUT


def test_classify_blocked() -> None:
    assert classify_tool_outcome(
        "execute_command",
        {"ok": False, "blocked": True, "stderr": "outside workspace"},
    ) == STATUS_BLOCKED


def test_classify_web_failure() -> None:
    assert classify_tool_outcome(
        "search_web",
        {"ok": False, "error": "Web search failed"},
    ) == STATUS_EXTERNAL_FAILURE


def test_classify_success() -> None:
    assert classify_tool_outcome(
        "read_file",
        {"ok": True, "content": "hello"},
    ) == STATUS_SUCCESS


def test_recovery_guidance_mentions_discovery_for_missing_target() -> None:
    guidance = recovery_guidance("read_file", STATUS_NOT_FOUND)
    assert "search_files" in guidance
    assert "list_directory" in guidance


def test_classify_file_already_exists_as_invalid_input() -> None:
    assert classify_tool_outcome(
        "file_mutation",
        {"ok": False, "error": "File already exists: hello.txt"},
    ) == STATUS_INVALID_INPUT
