from pathlib import Path

from agent.memory import MemoryStore


def test_memory_store_persists_entries(tmp_path: Path) -> None:
    path = tmp_path / "memory.json"

    store = MemoryStore(path)
    entry = store.add(
        "ユーザーはローカルAIを好む",
        tags=["preference", "local"],
    )

    restored = MemoryStore(path)

    assert restored.all()[0].id == entry.id
    assert restored.all()[0].content == "ユーザーはローカルAIを好む"
    assert restored.all()[0].tags == ["preference", "local"]


def test_memory_search_returns_relevant_entries(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory.json")
    store.add("WPF project uses PostgreSQL and PostGIS", ["project"])
    store.add("Bouldering is usually on weekends", ["hobby"])

    results = store.search("PostGIS")

    assert len(results) == 1
    assert "PostgreSQL" in results[0].content


def test_memory_search_respects_limit(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory.json")
    for index in range(5):
        store.add(f"local agent memory {index}")

    assert len(store.search("local agent", limit=2)) == 2


def test_memory_delete(tmp_path: Path) -> None:
    path = tmp_path / "memory.json"
    store = MemoryStore(path)
    entry = store.add("temporary")

    assert store.delete(entry.id) is True
    assert store.all() == []
    assert store.delete(entry.id) is False
