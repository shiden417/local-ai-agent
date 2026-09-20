from pathlib import Path
from typing import Any

from tools.path_utils import resolve_workspace_path, to_workspace_relative


MAX_RESULTS = 50
MAX_FILE_SIZE = 1_000_000
IGNORED_DIRECTORIES = {
    ".git",
    ".venv",
    "__pycache__",
    "bin",
    "obj",
    "node_modules",
}


def _read_text(path: Path) -> str | None:
    try:
        if path.stat().st_size > MAX_FILE_SIZE:
            return None
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return path.read_text(encoding="cp932", errors="replace")
    except OSError:
        return None


def search_files(
    working_directory: Path,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    query = str(arguments.get("query", ""))
    if not query:
        return {"ok": False, "error": "query must not be empty"}

    relative_path = str(arguments.get("path", "."))
    root = resolve_workspace_path(working_directory, relative_path)

    if not root.exists():
        return {"ok": False, "error": f"Path does not exist: {relative_path}"}

    case_sensitive = bool(arguments.get("case_sensitive", False))
    needle = query if case_sensitive else query.lower()

    matches: list[dict[str, Any]] = []

    candidates = [root] if root.is_file() else root.rglob("*")
    for path in candidates:
        if len(matches) >= MAX_RESULTS:
            break
        if not path.is_file():
            continue
        if any(part in IGNORED_DIRECTORIES for part in path.parts):
            continue

        content = _read_text(path)
        if content is None:
            continue

        haystack = content if case_sensitive else content.lower()
        line_number = None
        for number, line in enumerate(content.splitlines(), start=1):
            line_haystack = line if case_sensitive else line.lower()
            if needle in line_haystack:
                line_number = number
                matches.append(
                    {
                        "path": to_workspace_relative(working_directory, path),
                        "line": line_number,
                        "text": line[:500],
                    }
                )
                break

    return {
        "ok": True,
        "query": query,
        "path": to_workspace_relative(working_directory, root)
        if root.is_dir()
        else to_workspace_relative(working_directory, root.parent),
        "matches": matches,
        "truncated": len(matches) >= MAX_RESULTS,
    }
