from pathlib import Path

from tools.create_file import create_file
from tools.list_directory import list_directory
from tools.read_file import read_file


def test_create_file_creates_new_text_file(tmp_path: Path) -> None:
    result = create_file(
        tmp_path,
        {
            "path": "test.html",
            "content": "<html><body>Hello</body></html>",
        },
    )

    assert result["ok"] is True
    assert result["created"] is True
    assert (tmp_path / "test.html").read_text(encoding="utf-8") == (
        "<html><body>Hello</body></html>"
    )


def test_create_file_does_not_overwrite_existing_file(tmp_path: Path) -> None:
    target = tmp_path / "test.html"
    target.write_text("original", encoding="utf-8")

    result = create_file(
        tmp_path,
        {"path": "test.html", "content": "replacement"},
    )

    assert result["ok"] is False
    assert "already exists" in result["error"]
    assert target.read_text(encoding="utf-8") == "original"


def test_file_tools_can_use_explicit_absolute_local_paths(
    tmp_path: Path,
) -> None:
    target_dir = tmp_path / "external"
    target_file = target_dir / "hello.txt"

    created = create_file(
        tmp_path,
        {
            "path": str(target_file),
            "content": "hello",
        },
    )

    listed = list_directory(
        tmp_path,
        {"path": str(target_dir)},
    )
    read = read_file(
        tmp_path,
        {"path": str(target_file)},
    )

    assert created["ok"] is True
    assert listed["ok"] is True
    assert listed["path"] == str(target_dir.resolve())
    assert listed["entries"][0]["name"] == "hello.txt"
    assert read["ok"] is True
    assert read["content"] == "1: hello"


def test_create_file_creates_parent_directories(tmp_path: Path) -> None:
    result = create_file(
        tmp_path,
        {
            "path": "nested/site/test.html",
            "content": "<h1>test</h1>",
        },
    )

    assert result["ok"] is True
    assert (tmp_path / "nested/site/test.html").exists()
