from pathlib import Path

from agent.capability_router import Capability
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
            capabilities=(Capability.MEMORY_READ,),
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

    excluded = registry.schemas_for(
        "前回の記憶を確認してください",
        excluded_tools={"memory"},
    )
    assert [schema["function"]["name"] for schema in excluded] == ["core"]

 
 
def test_registry_routes_default_capability_scopes() -> None:
    from agent.tools import create_default_tool_registry

    registry = create_default_tool_registry()

    assert registry.schemas_for("こんにちは") == []

    folder_tools = [
        schema["function"]["name"]
        for schema in registry.schemas_for(
            "このフォルダの一覧を確認してください"
        )
    ]
    assert folder_tools == ["list_directory", "read_file", "search_files"]

    process_tools = [
        schema["function"]["name"]
        for schema in registry.schemas_for("pytestを実行してください")
    ]
    assert process_tools == ["execute_command"]

    memory_tools = [
        schema["function"]["name"]
        for schema in registry.schemas_for("前回の記憶を確認してください")
    ]
    assert memory_tools == ["search_memory"]

    ambiguous_tools = [
        schema["function"]["name"]
        for schema in registry.schemas_for("どうすればよいですか")
    ]
    assert ambiguous_tools == []


def test_registry_exposes_read_and_write_tools_for_edit_tasks() -> None:
    from agent.tools import create_default_tool_registry

    registry = create_default_tool_registry()

    edit_tools = [
        schema["function"]["name"]
        for schema in registry.schemas_for("READMEを修正してください")
    ]

    assert edit_tools == [
        "list_directory",
        "read_file",
        "search_files",
        "create_file",
        "edit_file",
    ]


def test_registry_exposes_create_file_for_creation_request() -> None:
    from agent.tools import create_default_tool_registry

    registry = create_default_tool_registry()

    tools = [
        schema["function"]["name"]
        for schema in registry.schemas_for("HTMLファイルを作成してください")
    ]

    assert tools == [
        "list_directory",
        "read_file",
        "search_files",
        "create_file",
        "edit_file",
    ]
