from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


ToolHandler = Callable[[Path, dict[str, Any]], dict[str, Any]]


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler
    requires_confirmation: bool = False
    use_when: str = ""
    avoid_when: str = ""
    availability: str = "always"
    routing_hints: tuple[str, ...] = ()

    def is_candidate(self, task_text: str) -> bool:
        """Return whether this tool should be exposed for the current task."""
        if self.availability == "always":
            return True
        if self.availability != "on_demand":
            raise ValueError(
                f"Unsupported tool availability: {self.availability}"
            )

        normalized = task_text.casefold()
        return any(
            hint.casefold() in normalized
            for hint in self.routing_hints
            if hint.strip()
        )

    def schema(self) -> dict[str, Any]:
        description = self.description.strip()
        if self.use_when:
            description += f"\nWhen to use: {self.use_when.strip()}"
        if self.avoid_when:
            description += f"\nDo not use for: {self.avoid_when.strip()}"

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """Registry and dispatcher for agent capabilities."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    @property
    def schemas(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self._tools.values()]

    def schemas_for(
        self,
        task_text: str,
        excluded_tools: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Expose relevant capabilities while allowing runtime quarantine."""
        excluded = excluded_tools or set()
        return [
            tool.schema()
            for tool in self._tools.values()
            if tool.name not in excluded and tool.is_candidate(task_text)
        ]

    def get(self, name: str) -> ToolDefinition | None:
        return self._tools.get(name)

    def execute(
        self,
        name: str,
        arguments: dict[str, Any],
        working_directory: Path,
    ) -> dict[str, Any]:
        tool = self._tools.get(name)
        if tool is None:
            return {
                "ok": False,
                "error": f"Unknown tool: {name}",
            }

        try:
            return tool.handler(working_directory, arguments)
        except Exception as exc:
            return {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }

    def names(self) -> tuple[str, ...]:
        return tuple(self._tools.keys())
