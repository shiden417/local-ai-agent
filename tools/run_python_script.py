from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

from agent.observation import truncate_text


DEFAULT_TIMEOUT_SECONDS = 15
MAX_TIMEOUT_SECONDS = 30
MAX_SCRIPT_CHARS = 12_000
MAX_OUTPUT_CHARS = 8_000


def run_python_script(
    working_directory: Path,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """Run a temporary Python script in a controlled child process.

    This is an isolation boundary, not a security sandbox. The operation is
    always confirmation-gated by the ToolDefinition and is bounded by script
    size, timeout, and returned output size.
    """
    script = str(arguments.get("script", "")).strip()
    if not script:
        return {
            "ok": False,
            "error": "script must not be empty",
        }

    if len(script) > MAX_SCRIPT_CHARS:
        return {
            "ok": False,
            "error": f"script exceeds {MAX_SCRIPT_CHARS} characters",
        }

    try:
        timeout_seconds = int(
            arguments.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS)
        )
    except (TypeError, ValueError):
        return {
            "ok": False,
            "error": "timeout_seconds must be an integer",
        }

    if not 1 <= timeout_seconds <= MAX_TIMEOUT_SECONDS:
        return {
            "ok": False,
            "error": (
                f"timeout_seconds must be between 1 and "
                f"{MAX_TIMEOUT_SECONDS}"
            ),
        }

    cwd = Path(working_directory).resolve()
    if not cwd.exists() or not cwd.is_dir():
        return {
            "ok": False,
            "error": f"作業ディレクトリが存在しません: {cwd}",
        }

    environment = os.environ.copy()
    for key in (
        "PYTHONPATH",
        "PYTHONHOME",
        "PYTHONSTARTUP",
        "PYTHONUSERBASE",
    ):
        environment.pop(key, None)

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".py",
            prefix=".agent-script-",
            delete=False,
        ) as handle:
            handle.write(script)
            temporary_path = Path(handle.name)

        process = subprocess.Popen(
            [
                sys.executable,
                "-I",
                str(temporary_path),
            ],
            cwd=str(cwd),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            _terminate_process_tree(process.pid)
            stdout, stderr = process.communicate()
            stderr = (
                f"{stderr}\nスクリプトが{timeout_seconds}秒以内に終了しなかったため終了しました。"
            ).strip()
            return {
                "ok": False,
                "exit_code": -1,
                "stdout": _bound_output(stdout),
                "stderr": _bound_output(stderr),
                "timed_out": True,
            }

        return {
            "ok": process.returncode == 0,
            "exit_code": process.returncode,
            "stdout": _bound_output(stdout),
            "stderr": _bound_output(stderr),
            "timed_out": False,
        }
    except Exception as exc:
        return {
            "ok": False,
            "exit_code": -1,
            "stdout": "",
            "stderr": f"Pythonスクリプト実行中にエラーが発生しました: {exc}",
            "timed_out": False,
        }
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass


def _terminate_process_tree(pid: int) -> None:
    if sys.platform.startswith("win"):
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return

    try:
        os.kill(pid, 9)
    except OSError:
        pass


def _bound_output(text: str) -> str:
    bounded, _ = truncate_text(text, MAX_OUTPUT_CHARS)
    return bounded
