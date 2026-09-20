from pathlib import Path
from typing import Any

from tools.path_utils import resolve_workspace_path, to_workspace_relative


MAX_CHARS = 12_000


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="cp932", errors="replace")


def read_file(
    working_directory: Path,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    relative_path = str(arguments.get("path", ""))
    path = resolve_workspace_path(working_directory, relative_path)

    if not path.exists():
        return {"ok": False, "error": f"File does not exist: {relative_path}"}

    if not path.is_file():
        return {"ok": False, "error": f"Not a file: {relative_path}"}

    try:
        content = _read_text(path)
    except OSError as exc:
        return {"ok": False, "error": f"Unable to read file: {exc}"}

    lines = content.splitlines()
    start_line = max(1, int(arguments.get("start_line", 1)))
    end_line = arguments.get("end_line")

    if end_line is None:
        selected = lines[start_line - 1 :]
    else:
        selected = lines[start_line - 1 : max(start_line - 1, int(end_line))]

    numbered_lines = [
        f"{number}: {line}"
        for number, line in enumerate(
            selected,
            start=start_line,
        )
    ]
    output = "\n".join(numbered_lines)

    truncated = len(output) > MAX_CHARS
    if truncated:
        output = output[:MAX_CHARS]
        output = output.rsplit("\n", 1)[0]

    return {
        "ok": True,
        "path": to_workspace_relative(working_directory, path),
        "start_line": start_line,
        "end_line": end_line,
        "content": output,
        "truncated": truncated,
    }
