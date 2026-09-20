from types import SimpleNamespace

import tools.search_web as module


def test_search_web_bounds_results(monkeypatch) -> None:
    class FakeDDGS:
        def __init__(self, timeout):
            assert timeout == 10

        def text(self, **kwargs):
            assert kwargs["region"] == "jp-ja"
            return [
                {"title": "Result", "href": "https://example.com", "body": "Snippet"},
                {"title": "", "href": "", "body": ""},
            ]

    monkeypatch.setattr(module, "DDGS", FakeDDGS)

    result = module.search_web("Python", max_results=5)

    assert result["ok"] is True
    assert result["count"] == 1
    assert result["results"][0]["url"] == "https://example.com"


def test_search_web_rejects_invalid_input() -> None:
    assert module.search_web("")["ok"] is False
    assert module.search_web("x", max_results=0)["ok"] is False
    assert module.search_web("x", timelimit="hour")["ok"] is False


def test_search_web_handles_provider_error(monkeypatch) -> None:
    class BrokenDDGS:
        def __init__(self, timeout):
            pass

        def text(self, **kwargs):
            raise RuntimeError("backend unavailable")

    monkeypatch.setattr(module, "DDGS", BrokenDDGS)
    result = module.search_web("Python")

    assert result["ok"] is False
    assert "backend unavailable" in result["error"]
