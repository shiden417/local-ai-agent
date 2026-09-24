from pathlib import Path

from tools.create_file import create_file
from tools.delete_file import delete_file
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
    assert listed["path"] == "external"
    assert listed["entries"][0]["name"] == "hello.txt"
    assert read["ok"] is True
    assert read["content"] == "hello"
    assert read["numbered_content"] == "1: hello"


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


def test_delete_file_deletes_existing_file(tmp_path: Path) -> None:
    target = tmp_path / "test.html"
    target.write_text("hello", encoding="utf-8")

    result = delete_file(tmp_path, {"path": "test.html"})

    assert result["ok"] is True
    assert result["deleted"] is True
    assert not target.exists()


def test_delete_file_does_not_delete_directory(tmp_path: Path) -> None:
    target = tmp_path / "folder"
    target.mkdir()

    result = delete_file(tmp_path, {"path": "folder"})

    assert result["ok"] is False
    assert "Not a file" in result["error"]
    assert target.exists()


def test_delete_file_supports_explicit_absolute_local_path(tmp_path: Path) -> None:
    target = tmp_path / "external.html"
    target.write_text("hello", encoding="utf-8")

    result = delete_file(tmp_path, {"path": str(target)})

    assert result["ok"] is True
    assert result["deleted"] is True
    assert not target.exists()


def test_file_mutation_dispatches_create_edit_and_delete(tmp_path: Path) -> None:
    from tools.file_mutation import file_mutation

    created = file_mutation(
        tmp_path,
        {
            "operation": "create",
            "path": "site.html",
            "content": "<h1>old</h1>",
        },
    )
    assert created["ok"] is True

    edited = file_mutation(
        tmp_path,
        {
            "operation": "edit",
            "path": "site.html",
            "search_text": "<h1>old</h1>",
            "replace_text": "<h1>new</h1>",
        },
    )
    assert edited["ok"] is True

    deleted = file_mutation(
        tmp_path,
        {
            "operation": "delete",
            "path": "site.html",
        },
    )
    assert deleted["ok"] is True
    assert not (tmp_path / "site.html").exists()


def test_file_mutation_rejects_unknown_operation(tmp_path: Path) -> None:
    from tools.file_mutation import file_mutation

    result = file_mutation(
        tmp_path,
        {
            "operation": "rename",
            "path": "site.html",
        },
    )

    assert result["ok"] is False
    assert "Unsupported file mutation operation" in result["error"]


def test_create_file_requires_content_argument(tmp_path: Path) -> None:
    result = create_file(tmp_path, {"path": "empty.txt"})

    assert result["ok"] is False
    assert "content is required" in result["error"]
    assert not (tmp_path / "empty.txt").exists()



def test_list_directory_explains_file_targets(tmp_path: Path) -> None:
    target = tmp_path / "sample.py"
    target.write_text("print('ok')\n", encoding="utf-8")

    result = list_directory(tmp_path, {"path": "sample.py"})

    assert result["ok"] is False
    assert result["suggested_tool"] == "read_file"
