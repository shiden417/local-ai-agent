import json
from pathlib import Path
from types import SimpleNamespace

from tools.dogfood_benchmark import (
    _add_required_regression_test,
    _inject_truncation_bug,
    _successful_pytest_command,
    _task1_passed,
    _task2_passed,
)


def test_inject_truncation_bug_changes_only_expected_source(tmp_path: Path) -> None:
    observation = tmp_path / "agent" / "observation.py"
    observation.parent.mkdir()
    original = (
        "def truncate_text(text, max_chars=8000):\n"
        "    return text[:head] + marker + text[-tail:], True\n"
    )
    observation.write_text(original, encoding="utf-8")

    before, _ = _inject_truncation_bug(tmp_path)

    assert before == original
    assert observation.read_text(encoding="utf-8") == (
        "def truncate_text(text, max_chars=8000):\n"
        "    return text[:head] + marker + text[-head:], True\n"
    )


def test_add_required_regression_test_is_deterministic(tmp_path: Path) -> None:
    tests = tmp_path / "tests" / "test_observation.py"
    tests.parent.mkdir()
    original = "from agent.observation import truncate_text\n"
    tests.write_text(original, encoding="utf-8")

    result = _add_required_regression_test(tmp_path)

    content = tests.read_text(encoding="utf-8")
    assert result == content
    assert content.startswith(original)
    assert "test_truncate_text_respects_max_chars_for_small_limits" in content



def test_successful_pytest_command_ignores_other_successful_tools() -> None:
    runtime = SimpleNamespace(
        current_task=SimpleNamespace(
            messages=[
                {
                    "role": "tool",
                    "name": "run_python_script",
                    "content": json.dumps({
                        "ok": True,
                        "exit_code": 0,
                        "stderr": "No module named pytest",
                    }),
                },
                {
                    "role": "tool",
                    "name": "execute_command",
                    "content": json.dumps({
                        "ok": True,
                        "exit_code": 0,
                        "command": "python -c \"print('done')\"",
                        "stdout": "done",
                    }),
                },
            ]
        )
    )

    assert _successful_pytest_command(runtime) is False


def test_successful_pytest_command_requires_passing_output() -> None:
    runtime = SimpleNamespace(
        current_task=SimpleNamespace(
            messages=[
                {
                    "role": "tool",
                    "name": "execute_command",
                    "content": json.dumps({
                        "ok": True,
                        "exit_code": 0,
                        "command": "python -m pytest -q",
                        "stdout": "35 passed in 1.2s",
                    }),
                }
            ]
        )
    )

    assert _successful_pytest_command(runtime) is True



def test_task1_accepts_equivalent_repair_that_uses_tail(tmp_path: Path) -> None:
    from agent.observation import truncate_text

    observation = tmp_path / "agent" / "observation.py"
    observation.parent.mkdir()
    injected = (
        "def truncate_text(text, max_chars=8000):\n"
        "    tail = 4\n"
        "    return text[:2] + text[-2:]\n"
    )
    repaired = (
        "def truncate_text(text, max_chars=8000):\n"
        "    tail = 4\n"
        "    return text[:2] + text[len(text) - tail:]\n"
    )
    observation.write_text(repaired, encoding="utf-8")
    tests = tmp_path / "tests" / "test_observation.py"
    tests.parent.mkdir(exist_ok=True)
    original_tests = "original tests"
    tests.write_text(original_tests, encoding="utf-8")
    runtime = SimpleNamespace(
        current_task=SimpleNamespace(
            messages=[
                {
                    "role": "tool",
                    "name": "execute_command",
                    "content": json.dumps(
                        {
                            "ok": True,
                            "exit_code": 0,
                            "command": "python -m pytest -q",
                            "stdout": "10 passed",
                        }
                    ),
                }
            ]
        )
    )

    ok, details = _task1_passed(tmp_path, injected, original_tests, runtime)

    assert ok is True
    assert "bug_repaired=True" in details


def test_task2_rejects_existing_test_modification(tmp_path: Path) -> None:
    path = tmp_path / "tests" / "test_completion_verifier.py"
    path.parent.mkdir()
    original = "def test_existing():\n    assert 1 == 1\n"
    path.write_text(
        "def test_existing():\n    assert 1 == 2\n\n"
        "def test_completion_verifier_accepts_blocked_status():\n"
        "    assert 'blocked' == 'blocked'\n",
        encoding="utf-8",
    )
    runtime = SimpleNamespace(
        current_task=SimpleNamespace(
            messages=[
                {
                    "role": "tool",
                    "name": "execute_command",
                    "content": json.dumps(
                        {
                            "ok": True,
                            "exit_code": 0,
                            "command": "python -m pytest -q",
                            "stdout": "11 passed",
                        }
                    ),
                }
            ]
        )
    )

    ok, details = _task2_passed(tmp_path, original, runtime)

    assert ok is False
    assert "existing_tests_preserved=False" in details


def test_task2_accepts_append_only_blocked_regression_test(tmp_path: Path) -> None:
    path = tmp_path / "tests" / "test_completion_verifier.py"
    path.parent.mkdir()
    original = "def test_existing():\n    assert 1 == 1\n"
    path.write_text(
        original
        + "\n"
        + "def test_completion_verifier_accepts_blocked_status():\n"
        + "    result = {'completion_status': 'blocked', 'summary': 'blocked'}\n"
        + "    assert result['completion_status'] == 'blocked'\n",
        encoding="utf-8",
    )
    runtime = SimpleNamespace(
        current_task=SimpleNamespace(
            messages=[
                {
                    "role": "tool",
                    "name": "execute_command",
                    "content": json.dumps(
                        {
                            "ok": True,
                            "exit_code": 0,
                            "command": "python -m pytest -q",
                            "stdout": "11 passed",
                        }
                    ),
                }
            ]
        )
    )

    ok, details = _task2_passed(tmp_path, original, runtime)

    assert ok is True
    assert "existing_tests_preserved=True" in details


def test_dogfood_environment_validator_uses_current_interpreter() -> None:
    from tools.dogfood_benchmark import _validate_dogfood_environment

    _validate_dogfood_environment()
