from pathlib import Path

from agent.memory import MemoryStore
from tools.memory import save_memory, search_memory


def test_save_memory_tool(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory.json")

    result = save_memory(
        store,
        tmp_path,
        {"content": "remember this", "tags": ["test"]},
    )

    assert result["ok"] is True
    assert result["content"] == "remember this"


def test_search_memory_tool(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory.json")
    store.add("The project uses Qwen3", ["project"])

    result = search_memory(
        store,
        tmp_path,
        {"query": "Qwen3"},
    )

    assert result["ok"] is True
    assert result["results"][0]["content"] == "The project uses Qwen3"
