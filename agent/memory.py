from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


DEFAULT_MEMORY_PATH = Path.home() / ".local-ai-agent" / "memory.json"


@dataclass
class MemoryEntry:
    id: str
    content: str
    tags: list[str]
    created_at: str
    updated_at: str


class MemoryStore:
    """Small persistent local memory store with deterministic keyword search."""

    def __init__(self, path: str | Path = DEFAULT_MEMORY_PATH) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: list[MemoryEntry] = []
        self._load()

    def add(self, content: str, tags: list[str] | None = None) -> MemoryEntry:
        content = content.strip()
        if not content:
            raise ValueError("memory content must not be empty")

        now = _now()
        entry = MemoryEntry(
            id=str(uuid4()),
            content=content,
            tags=_normalize_tags(tags or []),
            created_at=now,
            updated_at=now,
        )
        self._entries.append(entry)
        self._save()
        return entry

    def search(self, query: str, limit: int = 5) -> list[MemoryEntry]:
        query = query.strip()
        if not query:
            return []

        if limit <= 0:
            return []

        query_terms = _terms(query)
        results: list[tuple[int, str, MemoryEntry]] = []

        for entry in self._entries:
            haystack = " ".join([entry.content, *entry.tags]).lower()
            score = sum(haystack.count(term) for term in query_terms)
            if score:
                results.append((score, entry.updated_at, entry))

        results.sort(key=lambda item: (-item[0], item[1]), reverse=False)
        return [entry for _, _, entry in results[:limit]]

    def all(self) -> list[MemoryEntry]:
        return list(self._entries)

    def delete(self, memory_id: str) -> bool:
        original_count = len(self._entries)
        self._entries = [
            entry for entry in self._entries if entry.id != memory_id
        ]

        if len(self._entries) == original_count:
            return False

        self._save()
        return True

    def _load(self) -> None:
        if not self.path.exists():
            return

        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return

        if not isinstance(raw, list):
            return

        entries: list[MemoryEntry] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            try:
                entries.append(MemoryEntry(**item))
            except TypeError:
                continue

        self._entries = entries

    def _save(self) -> None:
        payload = [asdict(entry) for entry in self._entries]
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(self.path)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_tags(tags: list[str]) -> list[str]:
    result: list[str] = []
    for tag in tags:
        normalized = tag.strip().lower()
        if normalized and normalized not in result:
            result.append(normalized)
    return result


def _terms(query: str) -> list[str]:
    terms = re.findall(r"[\w一-龯ぁ-んァ-ヶ]+", query.lower(), flags=re.UNICODE)
    return list(dict.fromkeys(term for term in terms if term))
