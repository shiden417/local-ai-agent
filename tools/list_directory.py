from pathlib import Path
from typing import Any

from tools.path_utils import resolve_workspace_path, to_display_path


MAX_ENTRIES = 200


def list_directory(
    working_directory: Path,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    requested_path = str(arguments.get("path", "."))
    directory = resolve_workspace_path(working_directory, requested_path)

    if not directory.exists():
        return {"ok": False, "error": f"Directory does not exist: {requested_path}"}

    if not directory.is_dir():
        return {"ok": False, "error": f"Not a directory: {requested_path}"}

    entries = sorted(
        directory.iterdir(),
        key=lambda item: (not item.is_dir(), item.name.lower()),
    )
    truncated = len(entries) > MAX_ENTRIES
    entries = entries[:MAX_ENTRIES]

    result_entries = []
    for entry in entries:
        item = {
            "name": entry.name,
            "type": "directory" if entry.is_dir() else "file",
        }
        if entry.is_file():
            try:
                item["size"] = entry.stat().st_size
            except OSError:
                pass
        result_entries.append(item)

    return {
        "ok": True,
        "path": to_display_path(working_directory, directory),
        "entries": result_entries,
        "truncated": truncated,
    }
