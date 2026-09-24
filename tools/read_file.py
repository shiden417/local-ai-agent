from pathlib import Path
from typing import Any

from tools.path_utils import resolve_workspace_path, to_display_path


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
    requested_path = str(arguments.get("path", ""))
    path = resolve_workspace_path(working_directory, requested_path)

    if not path.exists():
        return {"ok": False, "error": f"File does not exist: {requested_path}"}

    if not path.is_file():
        return {"ok": False, "error": f"Not a file: {requested_path}"}

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

    raw_output = "\n".join(selected)
    numbered_lines = [
        f"{number}: {line}"
        for number, line in enumerate(
            selected,
            start=start_line,
        )
    ]
    numbered_output = "\n".join(numbered_lines)

    truncated = len(raw_output) > MAX_CHARS
    if truncated:
        raw_output = raw_output[:MAX_CHARS]
        raw_output = raw_output.rsplit("\n", 1)[0]
        numbered_output = "\n".join(
            f"{number}: {line}"
            for number, line in enumerate(
                raw_output.splitlines(),
                start=start_line,
            )
        )

    display_path = to_display_path(working_directory, path)
    display_directory = to_display_path(working_directory, path.parent)

    return {
        "ok": True,
        "path": display_path,
        "directory": display_directory,
        "relative_reference_base": display_directory,
        "start_line": start_line,
        "end_line": end_line,
        # `content` is the exact source text so its contents can be copied safely
        # into file_mutation.search_text/replace_text. `numbered_content` is retained
        # separately for display/navigation and is never part of the source itself.
        "content": raw_output,
        "numbered_content": numbered_output,
        "truncated": truncated,
    }
