from __future__ import annotations

import ast
from pathlib import Path


def validate_python_syntax(path: Path, content: str) -> str | None:
    """Return a concise validation error when Python source would become invalid."""
    if path.suffix.lower() != ".py":
        return None

    try:
        ast.parse(content, filename=str(path))
    except SyntaxError as exc:
        location = f"line {exc.lineno}" if exc.lineno else "unknown line"
        detail = exc.msg or "invalid syntax"
        return f"Python syntax validation failed at {location}: {detail}"

    return None
