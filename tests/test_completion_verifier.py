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


def test_completion_verifier_detects_missing_requested_test_call(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "src" / "calculator.py").write_text(
        "def add(a, b):\n    return a + b\n\ndef subtract(a, b):\n    return a - b\n",
        encoding="utf-8",
    )
    (tmp_path / "tests" / "test_calculator.py").write_text(
        "from src.calculator import add, subtract\n\n"
        "def test_add():\n    assert add(2, 3) == 5\n",
        encoding="utf-8",
    )
    verifier = CompletionVerifier(tmp_path)
    task = SimpleNamespace(
        goal=(
            "calculator に subtract(a, b) を追加してください。"
            "src/calculator.py と tests/test_calculator.py を変更し、"
            "python -m pytest -q を実行してください。"
        ),
        messages=[
            {
                "role": "tool",
                "name": "file_mutation",
                "content": json.dumps({"ok": True, "path": "src/calculator.py"}),
            },
            {
                "role": "tool",
                "name": "file_mutation",
                "content": json.dumps({"ok": True, "path": "tests/test_calculator.py"}),
            },
            {
                "role": "tool",
                "name": "execute_command",
                "content": json.dumps(
                    {"ok": True, "exit_code": 0, "command": "python -m pytest -q"}
                ),
            },
        ],
    )

    gaps = verifier.requirement_gaps(task.goal, task.messages)

    assert gaps
    assert any("subtract" in gap for gap in gaps)


def test_completion_verifier_accepts_requested_symbol_and_test_call(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "src" / "calculator.py").write_text(
        "def add(a, b):\n    return a + b\n\ndef subtract(a, b):\n    return a - b\n",
        encoding="utf-8",
    )
    (tmp_path / "tests" / "test_calculator.py").write_text(
        "from src.calculator import add, subtract\n\n"
        "def test_add():\n    assert add(2, 3) == 5\n\n"
        "def test_subtract():\n    assert subtract(5, 3) == 2\n",
        encoding="utf-8",
    )
    verifier = CompletionVerifier(tmp_path)
    task = SimpleNamespace(
        goal=(
            "calculator に subtract(a, b) を追加してください。"
            "src/calculator.py と tests/test_calculator.py を変更し、"
            "python -m pytest -q を実行してください。"
        ),
        messages=[
            {
                "role": "tool",
                "name": "file_mutation",
                "content": json.dumps({"ok": True, "path": "src/calculator.py"}),
            },
            {
                "role": "tool",
                "name": "file_mutation",
                "content": json.dumps({"ok": True, "path": "tests/test_calculator.py"}),
            },
            {
                "role": "tool",
                "name": "execute_command",
                "content": json.dumps(
                    {"ok": True, "exit_code": 0, "command": "python -m pytest -q"}
                ),
            },
        ],
    )

    assert verifier.requirement_gaps(task.goal, task.messages) == []
