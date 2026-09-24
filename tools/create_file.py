from __future__ import annotations

from pathlib import Path
from typing import Any

from tools.source_validation import validate_python_syntax

from tools.path_utils import resolve_workspace_path, to_display_path


MAX_FILE_SIZE = 1_000_000


def create_file(
    working_directory: Path,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """Create a new UTF-8 text file without overwriting an existing file."""
    requested_path = str(arguments.get("path", "")).strip()
    content = str(arguments.get("content", ""))

    if not requested_path:
        return {"ok": False, "error": "path must not be empty"}

    path = resolve_workspace_path(working_directory, requested_path)

    if path.exists():
        return {
            "ok": False,
            "error": f"File already exists: {requested_path}",
        }

    if len(content.encode("utf-8")) > MAX_FILE_SIZE:
        return {"ok": False, "error": "File content is too large to create safely"}

    validation_error = validate_python_syntax(path, content)
    if validation_error is not None:
        return {
            "ok": False,
            "error": validation_error,
            "path": to_display_path(working_directory, path),
            "validation_failed": True,
        }

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="")
    except OSError as exc:
        return {"ok": False, "error": f"Unable to create file: {exc}"}

    return {
        "ok": True,
        "path": to_display_path(working_directory, path),
        "created": True,
        "size": path.stat().st_size,
    }
