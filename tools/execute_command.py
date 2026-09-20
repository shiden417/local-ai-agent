from pathlib import Path
import subprocess


def execute_command(
    command: str,
    working_directory: str | Path,
    timeout_seconds: int = 30,
) -> dict:
    """Execute a PowerShell command in the agent workspace."""
    cwd = Path(working_directory).resolve()

    if not cwd.exists():
        return {"exit_code": -1, "stdout": "", "stderr": f"作業ディレクトリが存在しません: {cwd}"}

    try:
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            encoding="cp932",
            errors="replace",
            timeout=timeout_seconds,
        )
        return {
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except subprocess.TimeoutExpired:
        return {
            "exit_code": -1,
            "stdout": "",
            "stderr": f"コマンドが{timeout_seconds}秒以内に終了しませんでした。",
        }
    except Exception as exc:
        return {
            "exit_code": -1,
            "stdout": "",
            "stderr": f"コマンド実行中にエラーが発生しました: {exc}",
        }
