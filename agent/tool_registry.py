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
    terminal_on_success: bool = False

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
        self._schema_cache: dict[tuple[bool, tuple[str, ...]], list[dict[str, Any]]] = {}

    def register(self, tool: ToolDefinition) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool
        self._schema_cache.clear()

    @property
    def schemas(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self._tools.values()]

    def schemas_for(
        self,
        excluded_tools: set[str] | None = None,
        include_control_tools: bool = False,
    ) -> list[dict[str, Any]]:
        """Return all registered Tools eligible for this Agent task."""
        excluded = tuple(sorted(excluded_tools or set()))
        key = (include_control_tools, excluded)
        cached = self._schema_cache.get(key)
        if cached is not None:
            return list(cached)

        excluded_set = set(excluded)
        control_tools = {"ask_user", "finish_task"}
        schemas = [
            tool.schema()
            for tool in self._tools.values()
            if tool.name not in excluded_set
            and (
                include_control_tools
                or tool.name not in control_tools
            )
        ]
        self._schema_cache[key] = schemas
        return list(schemas)

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
