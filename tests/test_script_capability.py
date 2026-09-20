from agent.capability_router import Capability, CapabilityRouter, RoutingMode
from agent.tools import create_default_tool_registry


def test_router_classifies_script_fallback_tasks() -> None:
    route = CapabilityRouter().route("PDFをCSVに変換して")

    assert route.mode is RoutingMode.SCOPED
    assert Capability.SCRIPT_EXECUTION in route.capabilities


def test_registry_exposes_script_runner_for_script_tasks_only() -> None:
    registry = create_default_tool_registry()

    script_tools = [
        schema["function"]["name"]
        for schema in registry.schemas_for("PDFをCSVに変換して")
    ]
    assert script_tools == ["run_python_script"]

    conversational_tools = [
        schema["function"]["name"]
        for schema in registry.schemas_for("どうすればよいですか")
    ]
    assert conversational_tools == []
