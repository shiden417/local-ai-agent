from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from agent.completion_verifier import CompletionVerifier
from agent.runtime import AgentRuntime
from agent.task_requirements import classify_task_requirements


def _tool(name: str, payload: dict) -> dict:
    return {
        "role": "tool",
        "name": name,
        "content": json.dumps(payload, ensure_ascii=False),
    }


def test_task_requirements_keep_memory_separate_from_file_mutation() -> None:
    req = classify_task_requirements(
        "jarvis-benchmark をMemoryに保存し、その後検索して確認してください。"
    )

    assert req.file_mutation is False
    assert req.process_execution is False


def test_task_requirements_detect_explicit_file_save() -> None:
    req = classify_task_requirements(
        "results.txt に実行結果を保存してください。"
    )

    assert req.file_mutation is True


def test_runtime_allows_corrected_retry_for_invalid_input() -> None:
    runtime = AgentRuntime(Path("."))
    runtime.task = SimpleNamespace(
        recovery_tool="execute_command",
        last_failure_status="invalid_input",
    )

    assert runtime._can_retry_recovery_tool("execute_command") is True


def test_runtime_does_not_allow_same_tool_retry_for_not_found() -> None:
    runtime = AgentRuntime(Path("."))
    runtime.task = SimpleNamespace(
        recovery_tool="execute_command",
        last_failure_status="not_found",
    )

    assert runtime._can_retry_recovery_tool("execute_command") is False


def test_runtime_requires_latest_command_to_succeed() -> None:
    messages = [
        _tool(
            "execute_command",
            {"ok": True, "exit_code": 0, "command": "python -c \"print(1)\""},
        ),
        _tool(
            "execute_command",
            {"ok": True, "exit_code": 1, "command": "python -c \"raise SystemExit(1)\""},
        ),
    ]

    assert AgentRuntime._has_successful_command_execution(messages) is False


def test_runtime_accepts_latest_successful_command() -> None:
    messages = [
        _tool(
            "execute_command",
            {"ok": True, "exit_code": 1, "command": "python -c \"raise SystemExit(1)\""},
        ),
        _tool(
            "execute_command",
            {"ok": True, "exit_code": 0, "command": "python -c \"print(1)\""},
        ),
    ]

    assert AgentRuntime._has_successful_command_execution(messages) is True


def test_runtime_requires_latest_test_execution_to_succeed() -> None:
    messages = [
        _tool(
            "execute_command",
            {
                "ok": True,
                "exit_code": 0,
                "command": "python -m pytest -q",
                "stdout": "2 passed",
            },
        ),
        _tool(
            "execute_command",
            {
                "ok": True,
                "exit_code": 1,
                "command": "python -m pytest -q",
                "stdout": "1 failed, 1 passed",
            },
        ),
    ]

    assert AgentRuntime._has_successful_test_execution(messages) is False


def test_completion_verifier_rejects_prior_success_followed_by_failed_command(
    tmp_path: Path,
) -> None:
    verifier = CompletionVerifier(tmp_path)
    task = SimpleNamespace(
        goal=(
            "execute_command Toolを使い、python -c \"print('JARVIS benchmark')\" を実行し、"
            "終了コード0を確認してください。"
        ),
        messages=[
            _tool(
                "execute_command",
                {"ok": True, "exit_code": 0, "command": "python -c \"print(1)\""},
            ),
            _tool(
                "execute_command",
                {"ok": True, "exit_code": 1, "command": "python -c \"raise SystemExit(1)\""},
            ),
        ],
    )

    error = verifier.verify(
        task,
        {"completion_status": "completed"},
    )

    assert error is not None
    assert "exit_code 0" in error


def test_completion_verifier_rejects_prior_passing_test_followed_by_failed_test(
    tmp_path: Path,
) -> None:
    verifier = CompletionVerifier(tmp_path)
    task = SimpleNamespace(
        goal="calculator.pyを修正してpython -m pytest -qを実行し、全テストが成功することを確認してください。",
        messages=[
            _tool(
                "file_mutation",
                {"ok": True, "path": "calculator.py"},
            ),
            _tool(
                "execute_command",
                {
                    "ok": True,
                    "exit_code": 0,
                    "command": "python -m pytest -q",
                    "stdout": "2 passed",
                },
            ),
            _tool(
                "execute_command",
                {
                    "ok": True,
                    "exit_code": 1,
                    "command": "python -m pytest -q",
                    "stdout": "1 failed, 1 passed",
                },
            ),
        ],
    )

    error = verifier.verify(
        task,
        {"completion_status": "completed"},
    )

    assert error is not None
    assert "testing" in error.lower() or "pytest" in error.lower()


def test_completion_verifier_accepts_latest_successful_test(
    tmp_path: Path,
) -> None:
    verifier = CompletionVerifier(tmp_path)
    task = SimpleNamespace(
        goal="calculator.pyを修正してpython -m pytest -qを実行し、全テストが成功することを確認してください。",
        messages=[
            _tool(
                "file_mutation",
                {"ok": True, "path": "calculator.py"},
            ),
            _tool(
                "execute_command",
                {
                    "ok": True,
                    "exit_code": 1,
                    "command": "python -m pytest -q",
                    "stdout": "1 failed",
                },
            ),
            _tool(
                "execute_command",
                {
                    "ok": True,
                    "exit_code": 0,
                    "command": "python -m pytest -q",
                    "stdout": "2 passed",
                },
            ),
        ],
    )

    assert verifier.verify(
        task,
        {"completion_status": "completed"},
    ) is None


def test_loop_guard_blocks_identical_tool_call_after_two_attempts() -> None:
    from agent.loop_guard import ToolLoopGuard

    guard = ToolLoopGuard(max_identical_calls=2)
    arguments = {"query": "needle"}

    assert guard.record("search_files", arguments) == 1
    assert guard.is_repetition("search_files", arguments) is False

    assert guard.record("search_files", arguments) == 2
    assert guard.is_repetition("search_files", arguments) is True
