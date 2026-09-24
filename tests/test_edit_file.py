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


def test_edit_file_rejects_invalid_python_before_writing(tmp_path: Path) -> None:
    target = tmp_path / "example.py"
    original = "value = 1\n"
    target.write_text(original, encoding="utf-8")

    result = edit_file(
        tmp_path,
        {
            "path": "example.py",
            "search_text": "value = 1",
            "replace_text": 'assert \\"broken',
        },
    )

    assert result["ok"] is False
    assert result["validation_failed"] is True
    assert "Python syntax validation failed" in result["error"]
    assert target.read_text(encoding="utf-8") == original



def test_edit_file_recovers_line_number_prefixes(tmp_path: Path) -> None:
    target = tmp_path / "sample.py"
    target.write_text("alpha = 1\nbeta = 2\n", encoding="utf-8")

    result = edit_file(
        tmp_path,
        {
            "path": "sample.py",
            "search_text": "10: alpha = 1\n11: beta = 2",
            "replace_text": "alpha = 3\nbeta = 4",
        },
    )

    assert result["ok"] is True
    assert result["search_text_recovered"] is True
    assert target.read_text(encoding="utf-8") == "alpha = 3\nbeta = 4\n"


def test_edit_file_recovers_literal_newline_escapes(tmp_path: Path) -> None:
    target = tmp_path / "sample.py"
    target.write_text("alpha = 1\nbeta = 2\n", encoding="utf-8")

    result = edit_file(
        tmp_path,
        {
            "path": "sample.py",
            "search_text": "alpha = 1\\\\nbeta = 2",
            "replace_text": "alpha = 3\nbeta = 4",
        },
    )

    assert result["ok"] is True
    assert result["search_text_recovered"] is True
