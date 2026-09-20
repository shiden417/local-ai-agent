import json
from pathlib import Path

from agent.experimental_tools import register_experimental_tools
from agent.plugin_manager import PluginManager, PluginValidationError
from agent.recipe_store import RecipeStore
from agent.tool_registry import ToolRegistry


PLUGIN_SOURCE = '''def run(arguments):
    value = int(arguments.get("value", 0))
    return {"ok": True, "value": value * 2}
'''


def manifest(name: str = "double_value") -> dict:
    return {
        "name": name,
        "version": "0.1.0",
        "description": "Double an integer.",
        "parameters": {
            "type": "object",
            "properties": {
                "value": {"type": "integer"},
            },
            "required": ["value"],
            "additionalProperties": False,
        },
        "capabilities": ["script_execution"],
        "requires_confirmation": True,
        "use_when": "Double a supplied integer.",
        "avoid_when": "No numeric transformation is required.",
    }


def test_plugin_manager_stages_and_promotes(tmp_path: Path) -> None:
    manager = PluginManager(tmp_path / "plugins")

    staged = manager.stage("double-value", manifest(), PLUGIN_SOURCE)
    assert staged["ok"] is True
    assert staged["status"] == "quarantined"

    promoted = manager.promote("double-value")
    assert promoted["ok"] is True
    assert promoted["status"] == "enabled"
    assert (tmp_path / "plugins" / "enabled" / "double-value" / "plugin.py").exists()


def test_plugin_manager_rejects_invalid_source(tmp_path: Path) -> None:
    manager = PluginManager(tmp_path / "plugins")

    try:
        manager.stage(
            "broken",
            manifest("broken"),
            "def run(arguments):\n    return {",
        )
    except PluginValidationError as exc:
        assert "syntax error" in str(exc)
    else:
        raise AssertionError("invalid plugin source was accepted")


def test_plugin_manager_loads_and_executes_plugin(
    tmp_path: Path,
) -> None:
    manager = PluginManager(tmp_path / "plugins")
    manager.stage("double-value", manifest(), PLUGIN_SOURCE)
    manager.promote("double-value")

    registry = ToolRegistry()
    loaded = manager.load_enabled(registry)

    assert loaded[0]["tool_name"] == "double_value"
    assert registry.get("double_value") is not None
    assert registry.get("double_value").requires_confirmation is True
    result = registry.execute(
        "double_value",
        {"value": 21},
        tmp_path,
    )

    assert result == {"ok": True, "value": 42}


def test_registry_can_stage_and_promote_plugin(tmp_path: Path) -> None:
    from agent.tools import create_default_tool_registry

    manager = PluginManager(tmp_path / "plugins")
    recipes = RecipeStore(tmp_path / "recipes.json")
    registry = create_default_tool_registry()
    register_experimental_tools(registry, manager, recipes)

    staged = registry.execute(
        "stage_plugin",
        {
            "plugin_id": "double-value",
            "manifest": manifest(),
            "source": PLUGIN_SOURCE,
        },
        tmp_path,
    )
    assert staged["ok"] is True

    promoted = registry.execute(
        "promote_plugin",
        {"plugin_id": "double-value"},
        tmp_path,
    )
    assert promoted["ok"] is True
    assert promoted["registered"] is True
    assert registry.execute(
        "double_value",
        {"value": 7},
        tmp_path,
    ) == {"ok": True, "value": 14}

def test_plugin_manager_rejects_blocked_module(tmp_path: Path) -> None:
    manager = PluginManager(tmp_path / "plugins")
    blocked = "import subprocess\ndef run(arguments): return {'ok': True}"

    try:
        manager.stage(
            "blocked",
            manifest("blocked"),
            blocked,
        )
    except PluginValidationError as exc:
        assert "blocked module" in str(exc)
    else:
        raise AssertionError("blocked import was accepted")


def test_plugin_manager_rejects_blocked_builtin(tmp_path: Path) -> None:
    manager = PluginManager(tmp_path / "plugins")
    blocked = "def run(arguments): return open('secret.txt').read()"

    try:
        manager.stage(
            "blocked-builtin",
            manifest("blocked_builtin"),
            blocked,
        )
    except PluginValidationError as exc:
        assert "blocked builtin" in str(exc)
    else:
        raise AssertionError("blocked builtin was accepted")


def test_plugin_manager_tests_candidate_without_persisting(tmp_path: Path) -> None:
    manager = PluginManager(tmp_path / "plugins")
    result = manager.test_candidate(
        "double-value",
        manifest(),
        PLUGIN_SOURCE,
        {"value": 5},
        tmp_path,
    )

    assert result["ok"] is True
    assert result["status"] == "tested"
    assert not (tmp_path / "plugins" / "quarantine" / "double-value").exists()
