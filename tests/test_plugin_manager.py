import json
from pathlib import Path

from agent.capability_router import Capability, CapabilityRouter, RoutingMode
from agent.plugin_manager import PluginManager, PluginValidationError
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
        "capabilities": [Capability.SCRIPT_EXECUTION.value],
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


def test_plugin_router_exposes_capability_management() -> None:
    route = CapabilityRouter().route("新しいToolを追加してPDFを処理できるようにして")

    assert route.mode is RoutingMode.SCOPED
    assert (
        Capability.CAPABILITY_MANAGEMENT in route.capabilities
    )
