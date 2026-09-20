from pathlib import Path

from agent.recipe_store import RecipeStore


def test_recipe_store_records_and_persists(tmp_path: Path) -> None:
    path = tmp_path / "recipes.json"
    store = RecipeStore(path)

    first = store.record(
        "CSVをJSONに変換する",
        "print('convert')",
    )
    second = store.record(
        "同じCSV変換をもう一度する",
        "print('convert')",
    )

    assert first.id == second.id
    assert second.use_count == 2

    restored = RecipeStore(path)
    assert len(restored.all()) == 1
    assert restored.all()[0].use_count == 2


def test_recipe_store_search_and_promotion_candidates(tmp_path: Path) -> None:
    store = RecipeStore(tmp_path / "recipes.json")
    store.record("PDFの表をCSVへ変換", "print('pdf')")
    store.record("画像を解析", "print('image')")

    results = store.search("PDF CSV")
    assert [entry.goal for entry in results] == ["PDFの表をCSVへ変換"]

    store.record("PDFをもう一度変換", "print('pdf')")
    assert [entry.id for entry in store.promotion_candidates()] == [results[0].id]


def test_recipe_store_filters_promotion_candidates_by_use_count(tmp_path: Path) -> None:
    store = RecipeStore(tmp_path / "recipes.json")
    store.record("one use", "print('one')")

    assert store.promotion_candidates(min_uses=2) == []
