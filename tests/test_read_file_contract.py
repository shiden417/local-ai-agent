from pathlib import Path

from tools.read_file import read_file
from tools.edit_file import edit_file


def test_read_file_content_is_raw_source_and_numbered_content_is_separate(tmp_path: Path) -> None:
    path = tmp_path / "example.py"
    path.write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")

    result = read_file(
        tmp_path,
        {"path": "example.py"},
    )

    assert result["ok"] is True
    assert result["content"] == "def add(a, b):\n    return a + b"
    assert result["numbered_content"] == "1: def add(a, b):\n2:     return a + b"


def test_edit_file_strips_read_file_prefixes_from_replacement_after_recovery(tmp_path: Path) -> None:
    path = tmp_path / "session.txt"
    path.write_text("JARVIS SESSION\n", encoding="utf-8")

    result = edit_file(
        tmp_path,
        {
            "operation": "edit",
            "path": "session.txt",
            "search_text": "1: JARVIS SESSION",
            "replace_text": "1: JARVIS SESSION\n2: Session Context works",
        },
    )

    assert result["ok"] is True
    assert result["search_text_recovered"] is True
    assert path.read_text(encoding="utf-8") == "JARVIS SESSION\nSession Context works"
