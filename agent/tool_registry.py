from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path
from typing import Any, Callable

from agent.capability_router import Capability, CapabilityRouter, RoutingMode


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
    capabilities: tuple[Capability, ...] = ()
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

    def __init__(
        self,
        capability_router: CapabilityRouter | None = None,
    ) -> None:
        self._tools: dict[str, ToolDefinition] = {}
        self.capability_router = capability_router or CapabilityRouter()

    def register(self, tool: ToolDefinition) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        if tool.availability not in {"always", "on_demand"}:
            raise ValueError(
                f"Unsupported tool availability: {tool.availability}"
            )
        self._tools[tool.name] = tool

    @property
    def schemas(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self._tools.values()]

    def schemas_for(
        self,
        task_text: str,
        excluded_tools: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Expose only relevant capability families to the LLM.

        This is a capability gate, not a decision about whether a tool should
        actually be used. Once a capability is exposed, the LLM chooses the
        concrete tool and decides whether to call it.
        """
        excluded = excluded_tools or set()
        route = self.capability_router.route(task_text)

        if route.mode == RoutingMode.DIRECT:
            return [
                tool.schema()
                for tool in self._tools.values()
                if tool.name not in excluded and tool.availability == "always"
            ]

        if route.mode == RoutingMode.OPEN:
            # Keep ambiguous or general conversation tool-free for now.
            # Concrete operational intent is routed through SCOPED mode.
            # This prevents an 8B model from inventing a reason to use an
            # unrelated local tool simply because it is available.
            return [
                tool.schema()
                for tool in self._tools.values()
                if tool.name not in excluded and tool.availability == "always"
            ]

        if Capability.WORKSPACE_WRITE in route.capabilities and not self._is_edit_intent(task_text):
            return [
                tool.schema()
                for tool in self._tools.values()
                if tool.name not in excluded
                and (
                    tool.availability == "always"
                    or tool.name == "file_mutation"
                )
            ]

        return [
            tool.schema()
            for tool in self._tools.values()
            if tool.name not in excluded
            and (
                tool.availability == "always"
                or any(
                    capability in route.capabilities
                    for capability in tool.capabilities
                )
            )
        ]

    @staticmethod
    def _is_edit_intent(task_text: str) -> bool:
        return bool(
            re.search(
                r"(編集|変更|修正|書き換え|書換え|追加|modify|edit|fix|change|update)",
                task_text,
                flags=re.IGNORECASE,
            )
        )

    def route_for(self, task_text: str):
        return self.capability_router.route(task_text)

    def capabilities_for(self, task_text: str) -> set[Capability]:
        return self.capability_router.detect(task_text)

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
