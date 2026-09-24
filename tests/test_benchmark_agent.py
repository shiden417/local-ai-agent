from __future__ import annotations

from pathlib import Path

from types import SimpleNamespace

from tools.benchmark_agent import (
    _check_exact,
    _seed_workspace,
    _task_process_execution_succeeded,
    _task_test_execution_succeeded,
)


def test_benchmark_workspace_seed_is_deterministic(tmp_path: Path) -> None:
    _seed_workspace(tmp_path)

    assert _check_exact(
        tmp_path / "calculator.py",
        "def add(a, b):\n"
        "    return a - b\n\n"
        "def multiply(a, b):\n"
        "    return a * b\n",
    )
    assert _check_exact(
        tmp_path / "test_calculator.py",
        "from calculator import add, multiply\n\n"
        "def test_add():\n"
        "    assert add(2, 3) == 5\n\n"
        "def test_multiply():\n"
        "    assert multiply(2, 3) == 6\n",
    )


def test_benchmark_result_helpers_read_current_task_messages() -> None:
    runtime = SimpleNamespace(
        current_task=SimpleNamespace(
            messages=[
                {
                    "role": "tool",
                    "name": "execute_command",
                    "content": '{"ok": true, "exit_code": 0, "stdout": "2 passed in 0.01s", "stderr": ""}',
                },
                {
                    "role": "tool",
                    "name": "execute_command",
                    "content": '{"ok": true, "exit_code": 0, "stdout": "JARVIS benchmark\\n", "stderr": ""}',
                },
            ]
        )
    )

    assert _task_test_execution_succeeded(runtime) is True
    assert _task_process_execution_succeeded(runtime) is True
