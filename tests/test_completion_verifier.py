import json
from pathlib import Path
from types import SimpleNamespace

from agent.completion_verifier import CompletionVerifier


def _task(tmp_path: Path, *tool_messages: dict) -> SimpleNamespace:
    return SimpleNamespace(goal="調査してください", messages=list(tool_messages))


def test_completion_verifier_rejects_finish_without_successful_action(tmp_path: Path) -> None:
    verifier = CompletionVerifier(tmp_path)
    result = {"completion_status": "completed", "summary": "done"}

    error = verifier.verify(_task(tmp_path), result)

    assert error is not None
    assert "no successful action" in error


def test_completion_verifier_checks_successful_command_exit_code(tmp_path: Path) -> None:
    verifier = CompletionVerifier(tmp_path)
    task = _task(
        tmp_path,
        {
            "role": "tool",
            "name": "execute_command",
            "content": json.dumps({"ok": True, "exit_code": 1}),
        },
    )

    assert verifier.verify(task, {"completion_status": "completed", "summary": "done"}) is not None


def test_completion_verifier_checks_created_file(tmp_path: Path) -> None:
    verifier = CompletionVerifier(tmp_path)
    task = _task(
        tmp_path,
        {
            "role": "tool",
            "name": "file_mutation",
            "content": json.dumps({"ok": True, "path": "created.txt"}),
        },
    )

    error = verifier.verify(
        task,
        {"completion_status": "completed", "summary": "created"},
    )

    assert error is not None
    assert "does not exist" in error

    (tmp_path / "created.txt").write_text("ok", encoding="utf-8")
    assert (
        verifier.verify(
            task,
            {"completion_status": "completed", "summary": "created"},
        )
        is None
    )


def test_completion_verifier_requires_diagnostic_action_for_project_investigation(
    tmp_path: Path,
) -> None:
    verifier = CompletionVerifier(tmp_path)
    task = SimpleNamespace(
        goal="プロジェクトの問題点を確認する",
        messages=[],
    )

    error = verifier.verify(
        task,
        {"completion_status": "completed", "summary": "調査した"},
    )

    assert error is not None
    assert "concrete diagnostic action" in error


def test_completion_verifier_accepts_successful_workspace_command_without_diagnostic(
    tmp_path: Path,
) -> None:
    verifier = CompletionVerifier(tmp_path)
    task = SimpleNamespace(
        goal="このworkspaceで python -c を使って JARVIS benchmark と表示し、終了コード0を確認してください。",
        messages=[
            {
                "role": "tool",
                "name": "execute_command",
                "content": json.dumps(
                    {
                        "ok": True,
                        "exit_code": 0,
                        "command": "python -c \"print('JARVIS benchmark')\"",
                    }
                ),
            }
        ],
    )

    assert verifier.verify(
        task,
        {
            "completion_status": "completed",
            "summary": "コマンド実行と終了コード0を確認しました",
        },
    ) is None


def test_completion_verifier_requires_pytest_after_python_mutation(tmp_path: Path) -> None:
    (tmp_path / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    verifier = CompletionVerifier(tmp_path)
    task = _task(
        tmp_path,
        {
            "role": "tool",
            "name": "file_mutation",
            "content": json.dumps({"ok": True, "path": "tests/test_example.py"}),
        },
    )

    error = verifier.verify(
        task,
        {"completion_status": "completed", "summary": "changed"},
    )

    assert error is not None
    assert "python -m pytest" in error


def test_completion_verifier_accepts_pytest_after_python_mutation(tmp_path: Path) -> None:
    (tmp_path / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    verifier = CompletionVerifier(tmp_path)
    task = _task(
        tmp_path,
        {
            "role": "tool",
            "name": "file_mutation",
            "content": json.dumps({"ok": True, "path": "tests/test_example.py"}),
        },
        {
            "role": "tool",
            "name": "execute_command",
            "content": json.dumps(
                {"ok": True, "exit_code": 0, "command": "python -m pytest -q"}
            ),
        },
    )

    assert verifier.verify(
        task,
        {"completion_status": "completed", "summary": "changed and tested"},
    ) is None
