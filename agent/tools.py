from pathlib import Path

from agent.capability_router import Capability
from agent.memory import MemoryStore
from agent.plugin_generator import PluginGenerationError, generate_plugin_candidate
from agent.plugin_manager import PluginManager, PluginValidationError
from agent.recipe_store import RecipeStore
from agent.tool_registry import ToolDefinition, ToolRegistry
from tools.file_mutation import file_mutation
from tools.execute_command import execute_command
from tools.list_directory import list_directory
from tools.memory import save_memory, search_memory
from tools.read_file import read_file
from tools.search_files import search_files
from tools.search_web import search_web
from tools.run_python_script import run_python_script


def create_default_tool_registry(
    memory_store: MemoryStore | None = None,
    plugin_manager: PluginManager | None = None,
    recipe_store: RecipeStore | None = None,
) -> ToolRegistry:
    """Create the default local capability set."""
    registry = ToolRegistry()
    memory = memory_store or MemoryStore()
    plugins = plugin_manager or PluginManager()
    recipes = recipe_store or RecipeStore()
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
            name="search_web",
            description=(
                "Search the live web for current information and return bounded "
                "title, URL, and snippet results. This is read-only."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The web search query.",
                    },
                    "max_results": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 8,
                        "description": "Number of results to return. Default 5.",
                    },
                    "region": {
                        "type": "string",
                        "description": "Search region/language such as jp-ja or us-en.",
                    },
                    "timelimit": {
                        "anyOf": [
                            {"type": "string", "enum": ["d", "w", "m", "y"]},
                            {"type": "null"},
                        ],
                        "description": "Optional time filter: day, week, month, or year.",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            handler=lambda working_directory, arguments: search_web(
                str(arguments.get("query", "")),
                int(arguments.get("max_results", 5)),
                str(arguments.get("region", "jp-ja")),
                arguments.get("timelimit"),
            ),
            use_when=(
                "Current external information is needed, or the user explicitly asks "
                "for web/internet search."
            ),
            avoid_when=(
                "The answer is already known from the task context or local workspace."
            ),
            availability="on_demand",
            capabilities=(Capability.WEB_SEARCH,),
        )
    )
    registry.register(
        ToolDefinition(
            name="list_promotion_candidates",
            description=(
                "List successful temporary Script Recipes that have been used "
                "often enough to be considered for persistent Plugin promotion."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "min_uses": {
                        "type": "integer",
                        "minimum": 2,
                        "maximum": 20,
                        "description": "Minimum successful uses required. Default is 2.",
                    }
                },
                "required": [],
                "additionalProperties": False,
            },
            handler=lambda working_directory, arguments: _list_promotion_candidates(
                recipes,
                arguments,
            ),
            use_when=(
                "You need to decide whether a repeatedly successful temporary "
                "Recipe should become a persistent Plugin."
            ),
            avoid_when=(
                "You are handling an ordinary task and do not need to manage "
                "Agent capabilities."
            ),
            availability="on_demand",
            capabilities=(Capability.CAPABILITY_MANAGEMENT,),
        )
    )

    registry.register(
        ToolDefinition(
            name="generate_plugin",
            description=(
                "Generate a persistent Plugin candidate from a successful Recipe. "
                "The candidate is returned for validation and testing; it is not enabled automatically."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "recipe_id": {
                        "type": "string",
                        "description": "Recipe id selected for promotion.",
                    }
                },
                "required": ["recipe_id"],
                "additionalProperties": False,
            },
            handler=lambda working_directory, arguments: _generate_plugin(
                recipes,
                arguments,
            ),
            use_when="A repeated Recipe should be converted into a reusable persistent capability.",
            avoid_when="The Recipe has not been identified as a promotion candidate or a temporary script is sufficient.",
            availability="on_demand",
            capabilities=(Capability.CAPABILITY_MANAGEMENT,),
        )
    )

    registry.register(
        ToolDefinition(
            name="test_plugin_candidate",
            description=(
                "Run a generated Plugin candidate in an isolated child process "
                "using its generated test arguments. This does not enable the Plugin."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "plugin_id": {"type": "string"},
                    "manifest": {"type": "object"},
                    "source": {"type": "string"},
                    "test_arguments": {"type": "object"},
                },
                "required": ["plugin_id", "manifest", "source", "test_arguments"],
                "additionalProperties": False,
            },
            handler=lambda working_directory, arguments: _test_plugin_candidate(
                plugins,
                working_directory,
                arguments,
            ),
            requires_confirmation=True,
            use_when="A newly generated Plugin candidate must be behaviorally checked before staging.",
            avoid_when="The candidate has not been generated or the user did not authorize executing generated code.",
            availability="on_demand",
            capabilities=(Capability.CAPABILITY_MANAGEMENT,),
        )
    )

    registry.register(
        ToolDefinition(
            name="stage_plugin",
            description=(
                "Stage a new local Agent Plugin in quarantine. "
                "The plugin is validated but not enabled until promoted."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "plugin_id": {
                        "type": "string",
                        "description": "Unique local plugin id.",
                    },
                    "manifest": {
                        "type": "object",
                        "description": "Plugin manifest. Must describe the tool schema and capability.",
                    },
                    "source": {
                        "type": "string",
                        "description": "Complete plugin.py source defining run(arguments).",
                    },
                },
                "required": ["plugin_id", "manifest", "source"],
                "additionalProperties": False,
            },
            handler=lambda working_directory, arguments: _stage_plugin(
                plugins,
                arguments,
            ),
            requires_confirmation=True,
            use_when="A new persistent capability should be created and placed into quarantine for promotion.",
            avoid_when="A temporary script or an existing Tool is sufficient.",
            availability="on_demand",
            capabilities=(Capability.CAPABILITY_MANAGEMENT,),
        )
    )

    registry.register(
        ToolDefinition(
            name="promote_plugin",
            description=(
                "Promote a quarantined Agent Plugin to the enabled local "
                "capability set and load it into the current Tool Registry."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "plugin_id": {
                        "type": "string",
                        "description": "Quarantined plugin id to promote.",
                    }
                },
                "required": ["plugin_id"],
                "additionalProperties": False,
            },
            handler=lambda working_directory, arguments: _promote_plugin(
                plugins,
                registry,
                arguments,
            ),
            requires_confirmation=True,
            use_when="A quarantined plugin has been reviewed and should become a persistent capability.",
            avoid_when="The plugin has not been staged or the user did not request a persistent capability.",
            availability="on_demand",
            capabilities=(Capability.CAPABILITY_MANAGEMENT,),
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

    plugins.load_enabled(registry)
    return registry


def _generate_plugin(
    store: RecipeStore,
    arguments: dict,
) -> dict:
    recipe_id = str(arguments.get("recipe_id", "")).strip()
    recipe = store.get(recipe_id)
    if recipe is None:
        return {"ok": False, "error": f"Recipe not found: {recipe_id}"}

    try:
        candidate = generate_plugin_candidate(recipe)
    except PluginGenerationError as exc:
        return {"ok": False, "error": str(exc)}

    return {
        "ok": True,
        "status": "generated",
        "recipe_id": recipe.id,
        "recipe_use_count": recipe.use_count,
        "candidate": candidate,
    }


def _test_plugin_candidate(
    manager: PluginManager,
    working_directory: Path,
    arguments: dict,
) -> dict:
    required = ("plugin_id", "manifest", "source", "test_arguments")
    missing = [key for key in required if key not in arguments]
    if missing:
        return {
            "ok": False,
            "error": f"candidate missing: {', '.join(missing)}",
        }

    try:
        result = manager.test_candidate(
            str(arguments["plugin_id"]),
            arguments["manifest"],
            str(arguments["source"]),
            arguments["test_arguments"],
            working_directory,
        )
    except (TypeError, ValueError) as exc:
        return {"ok": False, "error": str(exc)}

    return result


def _stage_plugin(
    manager: PluginManager,
    arguments: dict,
) -> dict:
    try:
        plugin_id = str(arguments.get("plugin_id", ""))
        manifest = arguments.get("manifest")
        source = str(arguments.get("source", ""))
        if not isinstance(manifest, dict):
            return {"ok": False, "error": "manifest must be an object"}
        return manager.stage(plugin_id, manifest, source)
    except PluginValidationError as exc:
        return {"ok": False, "error": str(exc)}


def _promote_plugin(
    manager: PluginManager,
    registry: ToolRegistry,
    arguments: dict,
) -> dict:
    plugin_id = str(arguments.get("plugin_id", "")).strip()
    if not plugin_id:
        return {"ok": False, "error": "plugin_id must not be empty"}

    try:
        manifest = manager.peek_manifest(plugin_id)
    except (OSError, ValueError, PluginValidationError) as exc:
        return {"ok": False, "error": str(exc)}

    tool_name = str(manifest.get("name", "")).strip()
    if registry.get(tool_name) is not None:
        return {
            "ok": False,
            "error": f"Tool name already registered: {tool_name}",
        }

    result = manager.promote(plugin_id)
    if not result.get("ok"):
        return result

    try:
        tool = manager.load_plugin(Path(result["path"]))
        registry.register(tool)
    except (OSError, ValueError, PluginValidationError) as exc:
        return {
            "ok": False,
            "error": f"Plugin was promoted but could not be loaded: {exc}",
            "plugin_id": plugin_id,
        }

    result["registered"] = True
    return result


def _list_promotion_candidates(
    store: RecipeStore,
    arguments: dict,
) -> dict:
    try:
        min_uses = int(arguments.get("min_uses", 2))
    except (TypeError, ValueError):
        return {"ok": False, "error": "min_uses must be an integer"}

    if not 2 <= min_uses <= 20:
        return {"ok": False, "error": "min_uses must be between 2 and 20"}

    entries = store.promotion_candidates(min_uses=min_uses)
    return {
        "ok": True,
        "min_uses": min_uses,
        "candidates": [
            {
                "id": entry.id,
                "goal": entry.goal,
                "use_count": entry.use_count,
                "script": entry.script,
            }
            for entry in entries[:5]
        ],
    }
