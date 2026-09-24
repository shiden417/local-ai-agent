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



def test_task_requirements_detect_japanese_no_change_without_verb_suffix() -> None:
    req = classify_task_requirements(
        'このworkspaceでファイルを変更せず、execute_command Toolを使って python -c "print(1)" を実行してください。'
    )

    assert req.file_mutation is False
    assert req.mutation_forbidden is True
    assert req.process_execution is True
    assert req.required_process_tool == "execute_command"


def test_task_requirements_ignore_negative_test_mentions() -> None:
    req = classify_task_requirements(
        "README.md に説明を追記してください。コード、テスト、config.json、app.pyは変更しないでください。"
    )

    assert req.file_mutation is True
    assert req.test_verification is False


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


def test_task_requirements_extract_protected_path() -> None:
    req = classify_task_requirements(
        "calculator.pyを修正して、python -m pytest -qを実行してください。"
        "test_calculator.py は変更しないでください。"
    )

    assert req.file_mutation is True
    assert req.mutation_forbidden is False
    assert req.protected_paths == ("test_calculator.py",)


def test_task_requirements_detect_global_file_mutation_forbidden() -> None:
    req = classify_task_requirements(
        "jarvis-v9-memoryをMemoryに保存し、workspaceのファイルは変更しないでください。"
    )

    assert req.mutation_forbidden is True
    assert req.file_mutation is False


def test_runtime_blocks_protected_mutation_path(tmp_path: Path) -> None:
    runtime = AgentRuntime(tmp_path)
    requirements = classify_task_requirements(
        "calculator.pyを修正してください。test_calculator.py は変更しないでください。"
    )

    assert runtime._is_protected_mutation_path(
        {"path": "test_calculator.py"},
        requirements,
    ) is True
    assert runtime._is_protected_mutation_path(
        {"path": "calculator.py"},
        requirements,
    ) is False


def test_completion_verifier_rejects_global_file_mutation(
    tmp_path: Path,
) -> None:
    verifier = CompletionVerifier(tmp_path)
    task = SimpleNamespace(
        goal="Memoryに保存してください。workspaceのファイルは変更しないでください。",
        messages=[
            _tool(
                "save_memory",
                {"ok": True, "content": "memory"},
            ),
            _tool(
                "file_mutation",
                {"ok": True, "path": "memory_log.txt"},
            ),
        ],
    )

    error = verifier.verify(
        task,
        {"completion_status": "completed"},
    )

    assert error is not None
    assert "prohibits workspace file changes" in error


def test_completion_verifier_rejects_protected_mutation(
    tmp_path: Path,
) -> None:
    verifier = CompletionVerifier(tmp_path)
    task = SimpleNamespace(
        goal="calculator.pyを修正してください。test_calculator.py は変更しないでください。",
        messages=[
            _tool(
                "file_mutation",
                {"ok": True, "path": "test_calculator.py"},
            ),
        ],
    )

    error = verifier.verify(
        task,
        {"completion_status": "completed"},
    )

    assert error is not None
    assert "protected file" in error


def test_task_requirements_detect_explicit_execute_command() -> None:
    req = classify_task_requirements(
        'execute_command Toolを使って python -c "print(1)" を実行してください。'
    )

    assert req.process_execution is True
    assert req.required_process_tool == "execute_command"


def test_task_requirements_mark_process_command_as_execute_command() -> None:
    req = classify_task_requirements(
        'python -m pytest -q を実行して全テスト成功を確認してください。'
    )

    assert req.process_execution is True
    assert req.required_process_tool == "execute_command"
    assert req.file_mutation is False


def test_task_requirements_keep_plain_python_execution_unforced() -> None:
    req = classify_task_requirements("Pythonを実行してください。")

    assert req.process_execution is False
    assert req.required_process_tool is None


def test_completion_verifier_rejects_out_of_scope_mutation(
    tmp_path: Path,
) -> None:
    verifier = CompletionVerifier(tmp_path)
    task = SimpleNamespace(
        goal="python -c \"print('hello')\" を実行してください。",
        messages=[
            _tool(
                "file_mutation",
                {"ok": True, "path": "unexpected.txt"},
            ),
            _tool(
                "execute_command",
                {
                    "ok": True,
                    "exit_code": 0,
                    "command": "python -c \"print('hello')\"",
                },
            ),
        ],
    )

    error = verifier.verify(
        task,
        {"completion_status": "completed"},
    )

    assert error is not None
    assert "did not request a workspace file change" in error


def test_runtime_mutation_scope_excludes_file_tools_for_process_task(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured: list[set[str]] = []

    class Response:
        choices = [
            SimpleNamespace(
                message=SimpleNamespace(
                    role="assistant",
                    content="確認できました。",
                    tool_calls=[],
                )
            )
        ]

    def fake_ask_llm(_messages, tools=None):
        captured.append(
            {item["function"]["name"] for item in (tools or [])}
        )
        return Response()

    monkeypatch.setattr("agent.runtime.ask_llm", fake_ask_llm)

    runtime = AgentRuntime(tmp_path)
    runtime.run("python -c \"print('hello')\" を実行してください。")

    assert captured
    assert "file_mutation" not in captured[0]
    assert "execute_command" in captured[0]
