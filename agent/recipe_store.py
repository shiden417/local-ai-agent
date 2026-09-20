from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_RECIPE_PATH = Path.home() / ".local-ai-agent" / "recipes.json"
MAX_RECIPE_SCRIPT_CHARS = 12_000


@dataclass
class RecipeEntry:
    id: str
    goal: str
    script: str
    fingerprint: str
    use_count: int
    created_at: str
    updated_at: str


class RecipeStore:
    """Persistent local store for successful temporary scripts."""

    def __init__(self, path: str | Path = DEFAULT_RECIPE_PATH) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: list[RecipeEntry] = []
        self._load()

    def record(self, goal: str, script: str) -> RecipeEntry:
        goal = goal.strip()
        script = script.strip()
        if not goal:
            raise ValueError("recipe goal must not be empty")
        if not script:
            raise ValueError("recipe script must not be empty")
        if len(script) > MAX_RECIPE_SCRIPT_CHARS:
            raise ValueError(
                f"recipe script exceeds {MAX_RECIPE_SCRIPT_CHARS} characters"
            )

        fingerprint = _fingerprint(script)
        now = _now()

        for entry in self._entries:
            if entry.fingerprint == fingerprint:
                entry.goal = goal
                entry.use_count += 1
                entry.updated_at = now
                self._save()
                return entry

        entry = RecipeEntry(
            id=fingerprint[:16],
            goal=goal,
            script=script,
            fingerprint=fingerprint,
            use_count=1,
            created_at=now,
            updated_at=now,
        )
        self._entries.append(entry)
        self._save()
        return entry

    def search(self, query: str, limit: int = 3) -> list[RecipeEntry]:
        query = query.strip()
        if not query or limit <= 0:
            return []

        terms = _terms(query)
        if not terms:
            return []

        scored: list[tuple[int, str, RecipeEntry]] = []
        for entry in self._entries:
            haystack = f"{entry.goal} {entry.script}".lower()
            score = sum(haystack.count(term) for term in terms)
            if score:
                scored.append((score, entry.updated_at, entry))

        scored.sort(key=lambda item: (-item[0], item[1]), reverse=False)
        return [entry for _, _, entry in scored[:limit]]

    def promotion_candidates(self, min_uses: int = 2) -> list[RecipeEntry]:
        if min_uses < 1:
            raise ValueError("min_uses must be at least 1")
        return sorted(
            (entry for entry in self._entries if entry.use_count >= min_uses),
            key=lambda entry: (-entry.use_count, entry.updated_at),
        )

    def all(self) -> list[RecipeEntry]:
        return list(self._entries)

    def _load(self) -> None:
        if not self.path.exists():
            return

        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return

        if not isinstance(raw, list):
            return

        entries: list[RecipeEntry] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            try:
                entries.append(RecipeEntry(**item))
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


def _fingerprint(script: str) -> str:
    normalized = "\n".join(line.rstrip() for line in script.strip().splitlines())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _terms(query: str) -> list[str]:
    terms = re.findall(r"[\w一-龯ぁ-んァ-ヶ]+", query.lower(), flags=re.UNICODE)
    return list(dict.fromkeys(term for term in terms if term))
