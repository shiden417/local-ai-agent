from pathlib import Path

from tools.execute_command import execute_command


def test_execute_command_uses_working_directory(tmp_path: Path) -> None:
    result = execute_command(
        "Get-Location | Select-Object -ExpandProperty Path",
        working_directory=tmp_path,
    )

    assert result["exit_code"] == 0
    assert result["ok"] is True
    assert str(tmp_path.resolve()) in result["stdout"]


def test_execute_command_reports_missing_directory(tmp_path: Path) -> None:
    result = execute_command("Get-Date", working_directory=tmp_path / "missing")

    assert result["exit_code"] == -1
    assert result["ok"] is False
    assert "存在しません" in result["stderr"]


def test_execute_command_rejects_empty_command(tmp_path: Path) -> None:
    result = execute_command("  ", working_directory=tmp_path)

    assert result["exit_code"] == -1
    assert result["ok"] is False


def test_execute_command_truncates_large_output(tmp_path: Path) -> None:
    result = execute_command(
        "1..10000 | ForEach-Object { 'x' * 20 }",
        working_directory=tmp_path,
    )

    assert result["ok"] is True
    assert len(result["stdout"]) <= 8_000
