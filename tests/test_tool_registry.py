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
        )
    )

    assert registry.names() == ("hello",)
    assert registry.schemas[0]["type"] == "function"
    assert registry.schemas[0]["function"]["name"] == "hello"


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
