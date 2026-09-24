from __future__ import annotations

from pathlib import Path
from typing import Any

from tools.path_utils import resolve_workspace_path, to_display_path


MAX_FILE_SIZE = 1_000_000


def replace_line(
    working_directory: str | Path,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """Replace one 1-based line in a local text file."""
    requested_path = str(arguments.get("path", "")).strip()
    try:
        line_number = int(arguments.get("line_number"))
    except (TypeError, ValueError):
        return {"ok": False, "error": "line_number must be an integer"}

    new_text = str(arguments.get("new_text", ""))
    if not requested_path:
        return {"ok": False, "error": "path must not be empty"}
    if line_number < 1:
        return {"ok": False, "error": "line_number must be at least 1"}

    try:
        path = resolve_workspace_path(working_directory, requested_path)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}

    if not path.exists():
        return {"ok": False, "error": f"File does not exist: {requested_path}"}
    if not path.is_file():
        return {"ok": False, "error": f"Not a file: {requested_path}"}

    try:
        if path.stat().st_size > MAX_FILE_SIZE:
            return {"ok": False, "error": "File is too large to edit safely"}
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            content = path.read_text(encoding="cp932")
        except OSError as exc:
            return {"ok": False, "error": f"Unable to read file: {exc}"}
    except OSError as exc:
        return {"ok": False, "error": f"Unable to read file: {exc}"}

    lines = content.splitlines(keepends=True)
    if line_number > len(lines):
        return {
            "ok": False,
            "error": f"line_number {line_number} is outside the file (1-{len(lines)})",
        }

    original = lines[line_number - 1]
    newline = "\r\n" if original.endswith("\r\n") else "\n" if original.endswith("\n") else ""
    replacement = new_text.rstrip("\r\n") + newline
    if replacement == original:
        return {
            "ok": False,
            "error": "replacement would not change the file",
            "no_op": True,
            "path": to_display_path(working_directory, path),
        }

    lines[line_number - 1] = replacement
    new_content = "".join(lines)

    try:
        path.write_text(new_content, encoding="utf-8", newline="")
    except OSError as exc:
        return {"ok": False, "error": f"Unable to write file: {exc}"}

    return {
        "ok": True,
        "operation": "replace_line",
        "path": to_display_path(working_directory, path),
        "line_number": line_number,
        "old_text": original.rstrip("\r\n"),
        "new_text": new_text.rstrip("\r\n"),
    }
