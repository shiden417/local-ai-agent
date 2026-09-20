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
            description="List files and directories inside the current Agent workspace.",
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
            use_when="Discover the current workspace structure or the entries inside a known directory.",
            avoid_when="You need the contents of a specific file or need to search for text inside files.",
            availability="on_demand",
            routing_hints=(
                "フォルダ", "ディレクトリ", "一覧", "ファイル一覧", "構成",
                "何がある", "workspace", "folder", "directory", "list", "files",
            ),
        )
    )

    registry.register(
        ToolDefinition(
            name="read_file",
            description="Read a text file inside the current Agent workspace.",
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
            use_when="You already know which local file is relevant and need its contents.",
            avoid_when="You are only trying to discover which files exist, or you need to search unknown files for a specific text.",
            availability="on_demand",
            routing_hints=(
                "ファイル", "内容", "中身", "読んで", "読み取", "確認", "説明",
                "調べて", "file", "read", "content", "inspect", "explain",
            ),
        )
    )

    registry.register(
        ToolDefinition(
            name="search_files",
            description="Search for a specific text string inside local files in the current Agent workspace.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Specific text, symbol, identifier, or phrase to search for.",
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
            use_when="You know a concrete string or symbol to locate in file contents.",
            avoid_when="You are trying to find important files by role, filename, category, or vague natural-language descriptions.",
            availability="on_demand",
            routing_hints=(
                "検索", "探して", "どこに", "文字列", "シンボル", "検索して",
                "search", "find", "locate", "symbol", "text",
            ),
        )
    )

    registry.register(
        ToolDefinition(
            name="edit_file",
            description="Replace exactly one matching text block in a local text file.",
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
            use_when="The user explicitly wants a local file changed and you have already inspected the target content.",
            avoid_when="You have not read the target file yet or the user only asked for an explanation.",
            availability="on_demand",
            routing_hints=("編集", "変更", "修正", "書き換え", "更新", "追加", "modify", "edit", "change", "update", "fix"),
        )
    )

    registry.register(
        ToolDefinition(
            name="execute_command",
            description="Execute a PowerShell command in the current Agent workspace.",
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
            routing_hints=("実行", "コマンド", "テスト", "ビルド", "起動", "停止", "インストール", "git", "powershell", "run", "execute", "test", "build", "install"),
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
            routing_hints=("覚えて", "記憶", "メモリ", "保存して", "今後も", "覚えさせ", "remember", "memory", "save this", "for future"),
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
            routing_hints=("以前", "前回", "過去", "記憶", "覚えて", "メモリ", "覚えている", "remember", "previous", "past", "memory"),
        )
    )

    return registry
