from pathlib import Path

from tools.path_utils import resolve_workspace_path, to_display_path


def delete_file(
    working_directory: str | Path,
    arguments: dict,
) -> dict:
    """Delete one existing local file without deleting directories."""
    requested_path = str(arguments.get("path", "")).strip()

    try:
        target = resolve_workspace_path(working_directory, requested_path)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}

    if not target.exists():
        return {
            "ok": False,
            "error": f"File not found: {target}",
            "path": str(target),
        }

    if not target.is_file():
        return {
            "ok": False,
            "error": f"Not a file: {target}",
            "path": str(target),
        }

    try:
        target.unlink()
    except OSError as exc:
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "path": str(target),
        }

    return {
        "ok": True,
        "deleted": True,
        "path": to_display_path(working_directory, target),
    }
