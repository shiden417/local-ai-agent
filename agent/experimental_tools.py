from __future__ import annotations

from pathlib import Path
from typing import Any

from agent.plugin_generator import (
    PluginGenerationError,
    generate_plugin_candidate,
)
from agent.plugin_manager import PluginManager, PluginValidationError
from agent.recipe_store import RecipeStore
from agent.tool_registry import ToolDefinition, ToolRegistry


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


def register_experimental_tools(
    registry: ToolRegistry,
    plugins: PluginManager,
    recipes: RecipeStore,
) -> None:
    """Register optional Recipe/Plugin capabilities into a Tool Registry."""
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
        )
    )
    
    

    plugins.load_enabled(registry)
