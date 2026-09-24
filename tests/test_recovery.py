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


def test_invalid_file_edit_recovery_guidance_avoids_line_number_prefixes() -> None:
    guidance = recovery_guidance("file_mutation", STATUS_INVALID_INPUT)
    assert "search_text" in guidance
    assert "read_file line-number prefixes" in guidance



def test_classify_search_text_not_found_as_invalid_input() -> None:
    from agent.recovery import STATUS_INVALID_INPUT, classify_tool_outcome

    result = {"ok": False, "error": "search_text was not found"}

    assert classify_tool_outcome("file_mutation", result) == STATUS_INVALID_INPUT


def test_classify_file_target_directory_mistake_as_invalid_input() -> None:
    from agent.recovery import STATUS_INVALID_INPUT, classify_tool_outcome

    result = {"ok": False, "error": "Not a directory: agent/observation.py"}

    assert classify_tool_outcome("list_directory", result) == STATUS_INVALID_INPUT
