from pathlib import Path

from tools.run_python_script import run_python_script


def test_run_python_script_executes_in_child_process(tmp_path: Path) -> None:
    result = run_python_script(
        tmp_path,
        {"script": "print('hello from script')"},
    )

    assert result["ok"] is True
    assert result["exit_code"] == 0
    assert result["stdout"].strip() == "hello from script"
    assert result["timed_out"] is False


def test_run_python_script_rejects_empty_script(tmp_path: Path) -> None:
    result = run_python_script(tmp_path, {"script": "   "})

    assert result["ok"] is False
    assert "must not be empty" in result["error"]


def test_run_python_script_bounds_timeout(tmp_path: Path) -> None:
    result = run_python_script(
        tmp_path,
        {"script": "import time; time.sleep(2)", "timeout_seconds": 1},
    )

    assert result["ok"] is False
    assert result["timed_out"] is True
