from __future__ import annotations

from typing import Any

from ddgs import DDGS

from agent.observation import truncate_text


DEFAULT_REGION = "jp-ja"
DEFAULT_MAX_RESULTS = 5
MAX_RESULTS = 8
MAX_QUERY_CHARS = 500
MAX_SNIPPET_CHARS = 700
DEFAULT_TIMEOUT_SECONDS = 10


def search_web(
    query: str,
    max_results: int = DEFAULT_MAX_RESULTS,
    region: str = DEFAULT_REGION,
    timelimit: str | None = None,
) -> dict[str, Any]:
    query = str(query).strip()
    if not query:
        return {"ok": False, "error": "検索クエリを入力してください。"}
    if len(query) > MAX_QUERY_CHARS:
        return {
            "ok": False,
            "error": f"検索クエリは{MAX_QUERY_CHARS}文字以内で指定してください。",
        }

    try:
        max_results = int(max_results)
    except (TypeError, ValueError):
        return {"ok": False, "error": "max_results must be an integer"}
    if not 1 <= max_results <= MAX_RESULTS:
        return {
            "ok": False,
            "error": f"max_results must be between 1 and {MAX_RESULTS}",
        }

    region = str(region or DEFAULT_REGION).strip() or DEFAULT_REGION
    if timelimit not in {None, "d", "w", "m", "y"}:
        return {
            "ok": False,
            "error": "timelimit must be one of: d, w, m, y, or null",
        }

    try:
        results = DDGS(timeout=DEFAULT_TIMEOUT_SECONDS).text(
            query=query,
            region=region,
            safesearch="moderate",
            timelimit=timelimit,
            max_results=max_results,
            backend="auto",
        )
    except Exception as exc:
        return {
            "ok": False,
            "error": f"Web検索に失敗しました: {type(exc).__name__}: {exc}",
        }

    bounded_results: list[dict[str, str]] = []
    for item in results[:max_results]:
        title, _ = truncate_text(str(item.get("title", "")), 300)
        url, _ = truncate_text(str(item.get("href", "")), 1_000)
        snippet, _ = truncate_text(str(item.get("body", "")), MAX_SNIPPET_CHARS)
        if not title and not url and not snippet:
            continue
        bounded_results.append(
            {"title": title, "url": url, "snippet": snippet}
        )

    return {
        "ok": True,
        "query": query,
        "region": region,
        "timelimit": timelimit,
        "count": len(bounded_results),
        "results": bounded_results,
    }
