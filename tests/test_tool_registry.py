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


def test_registry_exposes_registered_tools_without_natural_language_routing() -> None:
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="read",
            description="Read",
            parameters={"type": "object", "properties": {}, "required": []},
            handler=lambda _working_directory, _arguments: {"ok": True},
        )
    )
    registry.register(
        ToolDefinition(
            name="write",
            description="Write",
            parameters={"type": "object", "properties": {}, "required": []},
            handler=lambda _working_directory, _arguments: {"ok": True},
        )
    )

    names = [schema["function"]["name"] for schema in registry.schemas_for()]
    assert names == ["read", "write"]


def test_registry_can_exclude_temporarily_unavailable_tools() -> None:
    registry = ToolRegistry()
    for name in ("first", "second"):
        registry.register(
            ToolDefinition(
                name=name,
                description=name,
                parameters={"type": "object", "properties": {}, "required": []},
                handler=lambda _working_directory, _arguments: {"ok": True},
            )
        )

    names = [schema["function"]["name"] for schema in registry.schemas_for(excluded_tools={"second"})]
    assert names == ["first"]


def test_registry_hides_control_tools_by_default() -> None:
    registry = ToolRegistry()
    for name in ("ask_user", "finish_task"):
        registry.register(
            ToolDefinition(
                name=name,
                description=name,
                parameters={"type": "object", "properties": {}, "required": []},
                handler=lambda _working_directory, _arguments: {"ok": True},
            )
        )

    assert registry.schemas_for() == []
    names = [schema["function"]["name"] for schema in registry.schemas_for(include_control_tools=True)]
    assert names == ["ask_user", "finish_task"]


def test_default_registry_contains_all_registered_tools() -> None:
    from agent.tools import create_default_tool_registry

    registry = create_default_tool_registry()
    names = [schema["function"]["name"] for schema in registry.schemas_for()]
    assert set(names) == set(registry.names()) - {"ask_user", "finish_task"}


def test_default_registry_exposes_controls_for_agent_tasks() -> None:
    from agent.tools import create_default_tool_registry

    registry = create_default_tool_registry()
    names = [schema["function"]["name"] for schema in registry.schemas_for(include_control_tools=True)]
    assert set(names) == set(registry.names())




def test_tool_registry_caches_schemas_until_registration_changes() -> None:
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="inspect",
            description="Inspect",
            parameters={"type": "object", "properties": {}, "required": []},
            handler=lambda _working_directory, _arguments: {"ok": True},
        )
    )

    first = registry.schemas_for()
    second = registry.schemas_for()
    assert first == second
    assert first is not second

    registry.register(
        ToolDefinition(
            name="write",
            description="Write",
            parameters={"type": "object", "properties": {}, "required": []},
            handler=lambda _working_directory, _arguments: {"ok": True},
        )
    )
    assert {item["function"]["name"] for item in registry.schemas_for()} == {"inspect", "write"}
