from pathlib import Path

from tools.execute_command import execute_command


def test_execute_command_uses_working_directory(tmp_path: Path) -> None:
    result = execute_command(
        "Get-Location | Select-Object -ExpandProperty Path",
        working_directory=tmp_path,
    )
    assert result["exit_code"] == 0
    assert str(tmp_path.resolve()) in result["stdout"]


def test_execute_command_reports_missing_directory(tmp_path: Path) -> None:
    result = execute_command("Get-Date", working_directory=tmp_path / "missing")
    assert result["exit_code"] == -1
    assert "存在しません" in result["stderr"]
