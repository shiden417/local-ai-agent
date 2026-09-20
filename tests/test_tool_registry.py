from pathlib import Path

from agent.tool_registry import ToolDefinition, ToolRegistry


def test_registry_exposes_openai_style_schema(tmp_path: Path) -> None:
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="hello",
            description="Say hello",
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
            },
            handler=lambda _working_directory, _arguments: {"ok": True},
            use_when="A greeting is needed.",
            avoid_when="No greeting is required.",
        )
    )

    assert registry.names() == ("hello",)
    assert registry.schemas[0]["type"] == "function"
    assert registry.schemas[0]["function"]["name"] == "hello"
    assert "When to use: A greeting is needed." in registry.schemas[0]["function"]["description"]
    assert "Do not use for: No greeting is required." in registry.schemas[0]["function"]["description"]


def test_registry_filters_on_demand_tools_by_task() -> None:
    registry = ToolRegistry()

    registry.register(
        ToolDefinition(
            name="core",
            description="Always available",
            parameters={"type": "object", "properties": {}, "required": []},
            handler=lambda _working_directory, _arguments: {"ok": True},
            availability="always",
        )
    )
    registry.register(
        ToolDefinition(
            name="memory",
            description="Memory operation",
            parameters={"type": "object", "properties": {}, "required": []},
            handler=lambda _working_directory, _arguments: {"ok": True},
            availability="on_demand",
            routing_hints=("記憶", "memory"),
        )
    )

    memory_schemas = registry.schemas_for("前回の記憶を確認してください")
    workspace_schemas = registry.schemas_for("このフォルダの中身を確認してください")

    assert [schema["function"]["name"] for schema in memory_schemas] == [
        "core",
        "memory",
    ]
    assert [schema["function"]["name"] for schema in workspace_schemas] == [
        "core",
    ]


def test_registry_dispatches_tool(tmp_path: Path) -> None:
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="hello",
            description="Say hello",
            parameters={"type": "object", "properties": {}, "required": []},
            handler=lambda _working_directory, _arguments: {"message": "hello"},
        )
    )

    assert registry.execute("hello", {}, tmp_path) == {"message": "hello"}


def test_registry_reports_unknown_tool(tmp_path: Path) -> None:
    registry = ToolRegistry()

    result = registry.execute("missing", {}, tmp_path)

    assert result["ok"] is False
    assert "Unknown tool" in result["error"]


def test_registry_can_mark_a_tool_as_confirmation_required(tmp_path: Path) -> None:
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="mutate",
            description="Mutate local state",
            parameters={"type": "object", "properties": {}, "required": []},
            handler=lambda _working_directory, _arguments: {"ok": True},
            requires_confirmation=True,
        )
    )

    assert registry.get("mutate") is not None
    assert registry.get("mutate").requires_confirmation is True
