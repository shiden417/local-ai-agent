from pathlib import Path

from agent.capability_router import Capability
from agent.memory import MemoryStore
from agent.tool_registry import ToolDefinition, ToolRegistry
from tools.file_mutation import file_mutation
from tools.execute_command import execute_command
from tools.list_directory import list_directory
from tools.memory import save_memory, search_memory
from tools.read_file import read_file
from tools.search_files import search_files
from tools.run_python_script import run_python_script


def create_default_tool_registry(
    memory_store: MemoryStore | None = None,
) -> ToolRegistry:
    """Create the default local capability set."""
    registry = ToolRegistry()
    memory = memory_store or MemoryStore()

    registry.register(
        ToolDefinition(
            name="list_directory",
            description="List files and directories in the Agent workspace or at an explicit local path.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Workspace-relative path, or an explicit absolute local directory path.",
                    }
                },
                "required": [],
                "additionalProperties": False,
            },
            handler=list_directory,
            use_when="Discover the current workspace structure or the entries inside a known directory.",
            avoid_when="You need the contents of a specific file or need to search for text inside files.",
            availability="on_demand",
            capabilities=(Capability.WORKSPACE_READ,),
        )
    )

    registry.register(
        ToolDefinition(
            name="read_file",
            description="Read a text file in the Agent workspace or at an explicit local path.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Workspace-relative file path, or an explicit absolute local file path.",
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
            use_when="You already know which local file is relevant and need its contents.",
            avoid_when="You are only trying to discover which files exist, or you need to search unknown files for a specific text.",
            availability="on_demand",
            capabilities=(Capability.WORKSPACE_READ,),
        )
    )

    registry.register(
        ToolDefinition(
            name="search_files",
            description="Search for a specific text string inside local files under the workspace or an explicit local path.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Specific text, symbol, identifier, or phrase to search for.",
                    },
                    "path": {
                        "type": "string",
                        "description": "Optional workspace-relative path, or an explicit absolute local path.",
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
            use_when="You know a concrete string or symbol to locate in file contents.",
            avoid_when="You are trying to find important files by role, filename, category, or vague natural-language descriptions.",
            availability="on_demand",
            capabilities=(Capability.WORKSPACE_READ,),
        )
    )

    registry.register(
        ToolDefinition(
            name="file_mutation",
            description=(
                "Perform one local file mutation: create, edit, or delete. "
                "Use one operation at a time and provide the fields required by "
                "that operation."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["create", "edit", "delete"],
                        "description": "Mutation to perform.",
                    },
                    "path": {
                        "type": "string",
                        "description": "Workspace-relative path or explicit absolute local file path.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Complete content. Required for create.",
                    },
                    "search_text": {
                        "type": "string",
                        "description": "Exact text to replace. Required for edit.",
                    },
                    "replace_text": {
                        "type": "string",
                        "description": "Replacement text. Required for edit.",
                    },
                },
                "required": ["operation", "path"],
                "additionalProperties": False,
            },
            handler=file_mutation,
            requires_confirmation=True,
            use_when=(
                "The user explicitly asks to create, edit, modify, change, "
                "or delete a local file."
            ),
            avoid_when=(
                "You only need to read/search files, or the user only wants "
                "an explanation."
            ),
            availability="on_demand",
            capabilities=(Capability.WORKSPACE_WRITE,),
            terminal_on_success=True,
        )
    )

    registry.register(
        ToolDefinition(
            name="run_python_script",
            description=(
                "Run a temporary Python script in the Agent workspace in a bounded "
                "child process. Use this as a fallback when no dedicated Tool can "
                "perform the requested computation or transformation."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "script": {
                        "type": "string",
                        "description": "Python source code to execute temporarily.",
                    },
                    "timeout_seconds": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 30,
                        "description": "Optional timeout in seconds. Default is 15.",
                    },
                },
                "required": ["script"],
                "additionalProperties": False,
            },
            handler=run_python_script,
            requires_confirmation=True,
            use_when=(
                "A small computation, transformation, parsing task, or other "
                "temporary Python capability is needed and no dedicated Tool exists."
            ),
            avoid_when=(
                "A dedicated Tool already represents the operation, or the task "
                "does not require actual local execution."
            ),
            availability="on_demand",
            capabilities=(Capability.SCRIPT_EXECUTION,),
        )
    )

    registry.register(
        ToolDefinition(
            name="execute_command",
            description=(
                "Execute a PowerShell command in the current Agent workspace. "
                "Keep file paths inside the workspace. For Python project tests, "
                "prefer 'python -m pytest' so the active project interpreter is used."
            ),
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
            use_when="An OS/process/automation operation is required and no more specific Tool exists.",
            avoid_when="A dedicated read, search, or edit Tool already represents the requested operation.",
            availability="on_demand",
            capabilities=(Capability.PROCESS,),
        )
    )

    registry.register(
        ToolDefinition(
            name="save_memory",
            description="Save a durable fact or preference for future tasks.",
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
            use_when="Information should survive the current task and be useful in future tasks.",
            avoid_when="You only need information from the current workspace or the current task's tool results.",
            availability="on_demand",
            capabilities=(Capability.MEMORY_WRITE,),
        )
    )

    registry.register(
        ToolDefinition(
            name="search_memory",
            description="Search durable local memory from previous tasks.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Concrete keywords describing the information you want to recall.",
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
            use_when="Past conversations or explicitly saved information are required for the current goal.",
            avoid_when="The answer can be obtained from the current workspace, current Tool results, or the user's current message.",
            availability="on_demand",
            capabilities=(Capability.MEMORY_READ,),
        )
    )

    return registry
