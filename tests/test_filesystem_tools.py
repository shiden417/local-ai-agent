from pathlib import Path

from tools.list_directory import list_directory
from tools.path_utils import resolve_workspace_path
from tools.read_file import read_file
from tools.search_files import search_files


def test_resolve_workspace_path_rejects_escape(tmp_path: Path) -> None:
    try:
        resolve_workspace_path(tmp_path, "../outside.txt")
    except ValueError as exc:
        assert "escapes" in str(exc)
    else:
        raise AssertionError("Expected workspace escape to be rejected")


def test_list_directory(tmp_path: Path) -> None:
    (tmp_path / "example.txt").write_text("hello", encoding="utf-8")

    result = list_directory(tmp_path, {"path": "."})

    assert result["ok"] is True
    assert result["entries"][0]["name"] == "example.txt"


def test_read_file_returns_raw_content_and_numbered_display(tmp_path: Path) -> None:
    (tmp_path / "example.txt").write_text("first\nsecond\n", encoding="utf-8")

    result = read_file(tmp_path, {"path": "example.txt"})

    assert result["ok"] is True
    assert result["content"] == "first\nsecond"
    assert result["numbered_content"] == "1: first\n2: second"


def test_search_files_finds_text(tmp_path: Path) -> None:
    (tmp_path / "example.txt").write_text("hello world\n", encoding="utf-8")

    result = search_files(tmp_path, {"query": "world"})

    assert result["ok"] is True
    assert result["matches"][0]["path"] == "example.txt"
    assert result["matches"][0]["line"] == 1
