from pathlib import Path

from agent.tool_registry import ToolDefinition, ToolRegistry
from tools.execute_command import execute_command
from tools.list_directory import list_directory
from tools.read_file import read_file
from tools.search_files import search_files


def create_default_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()

    registry.register(
        ToolDefinition(
            name="list_directory",
            description="List files and directories inside the Agent workspace. Use relative paths only.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative directory path. Use '.' for the workspace root.",
                    }
                },
                "required": [],
                "additionalProperties": False,
            },
            handler=list_directory,
        )
    )

    registry.register(
        ToolDefinition(
            name="read_file",
            description="Read a text file inside the Agent workspace. Use this before editing code.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative file path.",
                    },
                    "start_line": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Optional 1-based first line.",
                    },
                    "end_line": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Optional 1-based last line.",
                    },
                },
                "required": ["path"],
                "additionalProperties": False,
            },
            handler=read_file,
        )
    )

    registry.register(
        ToolDefinition(
            name="search_files",
            description="Search text inside files in the Agent workspace. Use this to find symbols, messages, or code references.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Text to search for.",
                    },
                    "path": {
                        "type": "string",
                        "description": "Optional relative path to a directory or file.",
                    },
                    "case_sensitive": {
                        "type": "boolean",
                        "description": "Whether the search should be case-sensitive.",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            handler=search_files,
        )
    )

    registry.register(
        ToolDefinition(
            name="execute_command",
            description="Execute a PowerShell command with the Agent workspace as its current directory.",
            parameters={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "PowerShell command to execute.",
                    }
                },
                "required": ["command"],
                "additionalProperties": False,
            },
            handler=lambda working_directory, arguments: execute_command(
                str(arguments.get("command", "")),
                working_directory=working_directory,
            ),
        )
    )

    return registry
