from __future__ import annotations

from pathlib import Path
import os
import re
import subprocess
import sys

from agent.safety import validate_command_scope
from agent.observation import truncate_text


DEFAULT_TIMEOUT_SECONDS = 30
MAX_OUTPUT_CHARS = 8_000


def execute_command(
    command: str,
    working_directory: str | Path,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> dict:
    """Execute a PowerShell command in the agent workspace."""
    if not command.strip():
        return {
            "ok": False,
            "exit_code": -1,
            "stdout": "",
            "command": command,
            "stderr": "command must not be empty",
        }

    cwd = Path(working_directory).resolve()

    if not cwd.exists():
        return {
            "ok": False,
            "exit_code": -1,
            "stdout": "",
            "command": command,
            "stderr": f"作業ディレクトリが存在しません: {cwd}",
        }

    scope_error = validate_command_scope(command, cwd)
    if scope_error is not None:
        return {
            "ok": False,
            "exit_code": -1,
            "stdout": "",
            "command": command,
            "stderr": scope_error,
            "timed_out": False,
            "blocked": True,
        }

    command = _normalize_python_command(command)

    wrapped_command = (
        "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; "
        "$OutputEncoding = [System.Text.Encoding]::UTF8; "
        "& { "
        f"{command}; "
        "if ($null -ne $LASTEXITCODE) { exit $LASTEXITCODE }; "
        "if (-not $?) { exit 1 } "
        "}"
    )

    environment = os.environ.copy()
    python_dir = str(Path(sys.executable).resolve().parent)
    current_path = environment.get("PATH", "")
    if python_dir not in current_path.split(os.pathsep):
        environment["PATH"] = (
            f"{python_dir}{os.pathsep}{current_path}"
            if current_path
            else python_dir
        )

    try:
        process = subprocess.Popen(
            [
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                wrapped_command,
            ],
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            _terminate_process_tree(process.pid)
            stdout, stderr = process.communicate()
            timeout_message = (
                f"コマンドが{timeout_seconds}秒以内に終了しなかったため終了しました。"
            )
            stderr = f"{stderr}\n{timeout_message}".strip()
            return {
                "ok": False,
                "exit_code": -1,
                "command": command,
                "stdout": _bound_output(stdout),
                "stderr": _bound_output(stderr),
                "timed_out": True,
            }

        return {
            "ok": process.returncode == 0,
            "exit_code": process.returncode,
            "command": command,
            "stdout": _bound_output(stdout),
            "stderr": _bound_output(stderr),
            "timed_out": False,
        }
    except Exception as exc:
        return {
            "ok": False,
            "exit_code": -1,
            "command": command,
            "stdout": "",
            "stderr": f"コマンド実行中にエラーが発生しました: {exc}",
            "timed_out": False,
        }


def _terminate_process_tree(pid: int) -> None:
    subprocess.run(
        ["taskkill", "/F", "/T", "/PID", str(pid)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def _bound_output(text: str) -> str:
    bounded, _ = truncate_text(text, MAX_OUTPUT_CHARS)
    return bounded



def _normalize_python_command(command: str) -> str:
    """Resolve Python/pytest commands to the interpreter running the Agent."""
    match = re.match(
        r"^(\s*)(python(?:\.exe)?)(?=\s|$)(.*)$",
        command,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match:
        prefix, _, rest = match.groups()
        return f'{prefix}& "{sys.executable}"{rest}'

    match = re.match(
        r"^(\s*)(pytest(?:\.exe)?)(?=\s|$)(.*)$",
        command,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match:
        prefix, _, rest = match.groups()
        return f'{prefix}& "{sys.executable}" -m pytest{rest}'

    return command
