from __future__ import annotations

from pathlib import Path
from typing import Any
import difflib
import re

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
    recovered = False

    if occurrence_count == 0:
        recovery_candidates = [
            _strip_read_file_line_numbers(search_text),
            _decode_literal_escapes(search_text),
            _strip_read_file_line_numbers(_decode_literal_escapes(search_text)),
        ]
        for candidate in recovery_candidates:
            if not candidate or candidate == search_text:
                continue
            candidate_count = content.count(candidate)
            if candidate_count == 1:
                search_text = candidate
                occurrence_count = 1
                recovered = True
                break

    if occurrence_count == 0:
        return {
            "ok": False,
            "error": (
                "search_text was not found. Use exact source text from the latest file contents; "
                'do not include read_file line-number prefixes such as "12: ".'
            ),
        }

    if occurrence_count > 1:
        return {
            "ok": False,
            "error": (
                f"search_text matched {occurrence_count} locations; provide a more specific block. "
                "Use surrounding lines (for example, the target function definition plus its assertion) "
                "so the next edit request is different from this one."
            ),
        }

    if recovered:
        normalized_replacement = _strip_read_file_line_numbers_if_present(replace_text)
        if normalized_replacement != replace_text:
            replace_text = normalized_replacement

    new_content = content.replace(search_text, replace_text, 1)

    if new_content == content:
        return {
            "ok": False,
            "error": (
                "edit would not change the file because search_text and replace_text "
                "produce identical content."
            ),
            "path": to_display_path(working_directory, path),
            "no_op": True,
        }

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
        "search_text_recovered": recovered,
    }



def _strip_read_file_line_numbers(value: str) -> str:
    lines = value.splitlines()
    if not lines or not all(line.lstrip().split(":", 1)[0].isdigit() and ":" in line for line in lines):
        return value
    return "\n".join(
        (
            rest[1:] if rest.startswith(" ") else rest
        )
        for rest in (line.split(":", 1)[1] for line in lines)
    )


def _strip_read_file_line_numbers_if_present(value: str) -> str:
    """Strip display-only read_file prefixes from model replacement text when present."""
    lines = value.splitlines()
    if len(lines) < 2 or not any(re.match(r"^\\d+: ", line) for line in lines):
        return value

    normalized: list[str] = []
    for line in lines:
        match = re.match(r"^\\d+: (.*)$", line)
        normalized.append(match.group(1) if match else line)
    return "\\n".join(normalized)


def _decode_literal_escapes(value: str) -> str:
    decoded = value
    # Models may serialize a newline as either \\n or \\\\n.
    while "\\\\n" in decoded:
        decoded = decoded.replace("\\\\n", "\\n")
    while "\\\\r" in decoded:
        decoded = decoded.replace("\\\\r", "\\r")
    return (
        decoded.replace("\\n", "\n")
        .replace("\\r", "\r")
        .replace('\\\"', '"')
    )
