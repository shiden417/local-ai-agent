from agent.request_classifier import RequestClassifier, RequestMode
from agent.tools import create_default_tool_registry

def test_request_classifier_marks_script_fallback_as_task() -> None:
    result = RequestClassifier().classify("PDFをCSVに変換して")
    assert result.mode is RequestMode.TASK


def test_registry_exposes_script_runner_without_capability_routing() -> None:
    registry = create_default_tool_registry()
    names = [
        schema["function"]["name"]
        for schema in registry.schemas_for()
    ]
    assert "run_python_script" in names
    assert "search_web" in names


def test_registry_does_not_decide_tool_scope_from_natural_language() -> None:
    registry = create_default_tool_registry()
    script_names = {schema["function"]["name"] for schema in registry.schemas_for()}
    conversational_names = {schema["function"]["name"] for schema in registry.schemas_for()}
    assert script_names == conversational_names
