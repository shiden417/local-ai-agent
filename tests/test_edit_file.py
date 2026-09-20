from pathlib import Path

from tools.edit_file import edit_file


def test_edit_file_replaces_exactly_one_match(tmp_path: Path) -> None:
    target = tmp_path / "example.txt"
    target.write_text("before\nmiddle\n", encoding="utf-8")

    result = edit_file(
        tmp_path,
        {
            "path": "example.txt",
            "search_text": "middle",
            "replace_text": "after",
        },
    )

    assert result["ok"] is True
    assert result["replacements"] == 1
    assert "middle" in result["diff"]
    assert target.read_text(encoding="utf-8") == "before\nafter\n"


def test_edit_file_rejects_missing_search_text(tmp_path: Path) -> None:
    target = tmp_path / "example.txt"
    target.write_text("hello\n", encoding="utf-8")

    result = edit_file(
        tmp_path,
        {
            "path": "example.txt",
            "search_text": "missing",
            "replace_text": "after",
        },
    )

    assert result["ok"] is False
    assert "not found" in result["error"]


def test_edit_file_rejects_ambiguous_match(tmp_path: Path) -> None:
    target = tmp_path / "example.txt"
    target.write_text("same\nsame\n", encoding="utf-8")

    result = edit_file(
        tmp_path,
        {
            "path": "example.txt",
            "search_text": "same",
            "replace_text": "changed",
        },
    )

    assert result["ok"] is False
    assert "2 locations" in result["error"]
