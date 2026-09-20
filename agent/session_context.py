from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

from agent.observation import truncate_text


MAX_TOPIC_CHARS = 300
MAX_GOAL_CHARS = 500
MAX_ANSWER_CHARS = 4_000
MAX_FACTS = 6
MAX_FACT_CHARS = 320
MAX_REFERENCES = 6


@dataclass
class SessionContext:
    """Small structured context retained across Tasks in one agent session."""

    current_topic: str = ""
    last_goal: str = ""
    last_answer: str = ""
    extracted_facts: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)

    def clear(self) -> None:
        self.current_topic = ""
        self.last_goal = ""
        self.last_answer = ""
        self.extracted_facts.clear()
        self.references.clear()

    def remember_task(
        self,
        goal: str,
        answer: str,
        messages: list[dict[str, Any]],
    ) -> None:
        self.current_topic, _ = truncate_text(
            str(goal).strip(),
            MAX_TOPIC_CHARS,
        )
        self.last_goal, _ = truncate_text(
            str(goal).strip(),
            MAX_GOAL_CHARS,
        )
        self.last_answer, _ = truncate_text(
            str(answer).strip(),
            MAX_ANSWER_CHARS,
        )

        facts: list[str] = []
        references: list[str] = []

        for message in reversed(messages):
            if message.get("role") != "tool":
                continue
            raw = str(message.get("content", ""))
            try:
                result = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(result, dict) or not result.get("ok"):
                continue

            name = str(message.get("name", ""))
            if name == "search_web":
                for item in result.get("results", [])[:3]:
                    if not isinstance(item, dict):
                        continue
                    title = str(item.get("title", "")).strip()
                    snippet = str(item.get("snippet", "")).strip()
                    if title or snippet:
                        fact = f"{title}: {snippet}".strip(": ")
                        fact, _ = truncate_text(fact, MAX_FACT_CHARS)
                        facts.append(fact)
                    url = str(item.get("url", "")).strip()
                    if _is_http_url(url):
                        references.append(url)

            elif name == "fetch_web_page":
                title = str(result.get("title", "")).strip()
                content = str(result.get("content", "")).strip()
                if title:
                    facts.append(f"Web page: {title}")
                if content:
                    first_sentences = " ".join(content.split())[:MAX_FACT_CHARS]
                    if first_sentences:
                        facts.append(f"Web content: {first_sentences}")
                url = str(result.get("final_url") or result.get("url", "")).strip()
                if _is_http_url(url):
                    references.append(url)

            elif name in {"file_mutation", "create_file", "edit_file", "delete_file"}:
                path = str(result.get("path", "")).strip()
                if path:
                    facts.append(f"Local file result: {path}")

        self.extracted_facts = _dedupe(facts)[:MAX_FACTS]
        self.references = _dedupe(references)[:MAX_REFERENCES]

    def prompt_block(self) -> str:
        if not self.current_topic and not self.last_answer:
            return (
                "[Session Context]\n"
                "No previous task context is retained for this session."
            )

        lines = ["[Session Context]"]
        if self.current_topic:
            lines.append(f"Current topic: {self.current_topic}")
        if self.last_goal:
            lines.append(f"Previous goal: {self.last_goal}")
        if self.extracted_facts:
            lines.append("Important retained facts:")
            lines.extend(f"- {fact}" for fact in self.extracted_facts)
        if self.last_answer:
            answer, _ = truncate_text(self.last_answer, 2_000)
            lines.append(f"Previous answer: {answer}")
        if self.references:
            lines.append("Relevant references:")
            lines.extend(f"- {url}" for url in self.references)
        return "\n".join(lines)


def _is_http_url(value: str) -> bool:
    try:
        return urlparse(value).scheme in {"http", "https"} and bool(
            urlparse(value).netloc
        )
    except ValueError:
        return False


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = value.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result
