from __future__ import annotations

from pathlib import Path
from typing import Any


def ask_user(
    _working_directory: Path,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    question = str(arguments.get("question", "")).strip()
    if not question:
        return {"ok": False, "error": "question must not be empty"}

    return {
        "ok": True,
        "status": "user_input_required",
        "question": question,
    }


def finish_task(
    _working_directory: Path,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    completion_status = str(arguments.get("completion_status", "")).strip().lower()
    summary = str(arguments.get("summary", "")).strip()

    if completion_status not in {"completed", "blocked"}:
        return {
            "ok": False,
            "error": "completion_status must be 'completed' or 'blocked'",
        }
    if not summary:
        return {"ok": False, "error": "summary must not be empty"}

    return {
        "ok": True,
        "completion_status": completion_status,
        "summary": summary,
    }
