from pathlib import Path
from typing import Any

from agent.memory import MemoryStore


def save_memory(
    store: MemoryStore,
    _working_directory: Path,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    content = str(arguments.get("content", ""))
    tags = arguments.get("tags", [])
    if not isinstance(tags, list):
        return {"ok": False, "error": "tags must be an array of strings"}
    if not all(isinstance(tag, str) for tag in tags):
        return {"ok": False, "error": "tags must contain only strings"}

    try:
        entry = store.add(content, tags)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}

    return {
        "ok": True,
        "id": entry.id,
        "content": entry.content,
        "tags": entry.tags,
    }


def search_memory(
    store: MemoryStore,
    _working_directory: Path,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    query = str(arguments.get("query", ""))
    limit = int(arguments.get("limit", 5))

    results = store.search(query, limit=limit)

    return {
        "ok": True,
        "query": query,
        "results": [
            {
                "id": entry.id,
                "content": entry.content,
                "tags": entry.tags,
                "updated_at": entry.updated_at,
            }
            for entry in results
        ],
        "hint": (
            "No matching memories were found. Do not repeat the same search; "
            "continue with the current task or use another Tool."
            if not results
            else None
        ),
    }
