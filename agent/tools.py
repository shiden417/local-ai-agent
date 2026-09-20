from pathlib import Path

from agent.memory import MemoryStore
from agent.tool_registry import ToolDefinition, ToolRegistry
from tools.edit_file import edit_file
from tools.execute_command import execute_command
from tools.list_directory import list_directory
from tools.memory import save_memory, search_memory
from tools.read_file import read_file
from tools.search_files import search_files


def create_default_tool_registry(
    memory_store: MemoryStore | None = None,
) -> ToolRegistry:
    """Create the default local capability set."""
    registry = ToolRegistry()
    memory = memory_store or MemoryStore()

    registry.register(
        ToolDefinition(
            name="list_directory",
            description="List files and directories inside the current Agent workspace. Use relative paths only.",
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
            description="Read a text file inside the current Agent workspace. Use this when you need local file contents.",
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
            description="Search text inside local files in the current Agent workspace.",
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
            name="edit_file",
            description="Replace exactly one matching text block in a local text file. Read the file first and provide an exact search_text block.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative file path.",
                    },
                    "search_text": {
                        "type": "string",
                        "description": "Exact text block to replace. It must occur exactly once.",
                    },
                    "replace_text": {
                        "type": "string",
                        "description": "Replacement text.",
                    },
                },
                "required": ["path", "search_text", "replace_text"],
                "additionalProperties": False,
            },
            handler=edit_file,
            requires_confirmation=True,
        )
    )

    registry.register(
        ToolDefinition(
            name="execute_command",
            description="Execute a PowerShell command in the current Agent workspace. Use this for OS operations or tasks without a dedicated Tool.",
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

    registry.register(
        ToolDefinition(
            name="save_memory",
            description="Save a durable fact or preference for future conversations. Use only for information that should survive the current task. Do not use this for current workspace contents. Do not store sensitive secrets or credentials.",
            parameters={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "A concise fact, preference, or durable piece of information to remember.",
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional short tags for later retrieval.",
                    },
                },
                "required": ["content"],
                "additionalProperties": False,
            },
            handler=lambda working_directory, arguments: save_memory(
                memory,
                working_directory,
                arguments,
            ),
            requires_confirmation=True,
        )
    )

    registry.register(
        ToolDefinition(
            name="search_memory",
            description="Search durable local memory for facts or preferences from previous tasks. Use this only to recall past information; do not use it to inspect the current workspace or current task files.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Keywords describing the information you want to recall.",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                        "description": "Maximum number of memory entries to return.",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            handler=lambda working_directory, arguments: search_memory(
                memory,
                working_directory,
                arguments,
            ),
        )
    )

    return registry
