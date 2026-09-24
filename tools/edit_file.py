from __future__ import annotations

from pathlib import Path
from typing import Any
import difflib

from tools.source_validation import validate_python_syntax

from tools.path_utils import resolve_workspace_path, to_display_path


MAX_FILE_SIZE = 1_000_000


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="cp932", errors="replace")


def _write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8", newline="")


def edit_file(
    working_directory: Path,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """Replace exactly one occurrence of search_text in a local text file."""
    requested_path = str(arguments.get("path", ""))
    search_text = str(arguments.get("search_text", ""))
    replace_text = str(arguments.get("replace_text", ""))

    if not search_text:
        return {"ok": False, "error": "search_text must not be empty"}

    path = resolve_workspace_path(working_directory, requested_path)

    if not path.exists():
        return {"ok": False, "error": f"File does not exist: {requested_path}"}

    if not path.is_file():
        return {"ok": False, "error": f"Not a file: {requested_path}"}

    try:
        if path.stat().st_size > MAX_FILE_SIZE:
            return {"ok": False, "error": "File is too large to edit safely"}
    except OSError as exc:
        return {"ok": False, "error": f"Unable to inspect file: {exc}"}

    try:
        content = _read_text(path)
    except OSError as exc:
        return {"ok": False, "error": f"Unable to read file: {exc}"}

    occurrence_count = content.count(search_text)

    if occurrence_count == 0:
        return {
            "ok": False,
            "error": (
                "search_text was not found. Use exact source text from the latest file contents; "
                "do not include read_file line-number prefixes such as \"12: \"."
            ),
        }

    if occurrence_count > 1:
        return {
            "ok": False,
            "error": f"search_text matched {occurrence_count} locations; provide a more specific block.",
        }

    new_content = content.replace(search_text, replace_text, 1)

    validation_error = validate_python_syntax(path, new_content)
    if validation_error is not None:
        return {
            "ok": False,
            "error": validation_error,
            "path": to_display_path(working_directory, path),
            "validation_failed": True,
        }

    try:
        _write_text(path, new_content)
    except OSError as exc:
        return {"ok": False, "error": f"Unable to write file: {exc}"}

    diff = "".join(
        difflib.unified_diff(
            content.splitlines(keepends=True),
            new_content.splitlines(keepends=True),
            fromfile=requested_path,
            tofile=requested_path,
        )
    )

    return {
        "ok": True,
        "path": to_display_path(working_directory, path),
        "replacements": 1,
        "diff": diff,
    }
