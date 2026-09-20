import json
from types import SimpleNamespace

import pytest

import agent.plugin_generator as generator
from agent.recipe_store import RecipeEntry


def recipe() -> RecipeEntry:
    return RecipeEntry(
        id="recipe-1",
        goal="数字を2倍にする",
        script="print('double')",
        fingerprint="abc",
        use_count=3,
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
    )


def test_generate_plugin_candidate_parses_model_json(monkeypatch) -> None:
    candidate = {
        "plugin_id": "double-value",
        "manifest": {
            "name": "double_value",
            "version": "0.1.0",
            "description": "Double a number.",
            "parameters": {
                "type": "object",
                "properties": {"value": {"type": "integer"}},
                "required": ["value"],
                "additionalProperties": False,
            },
            "capabilities": ["script_execution"],
            "use_when": "Double a number.",
            "avoid_when": "No numeric transformation.",
        },
        "source": "def run(arguments):\n    return {'ok': True, 'value': int(arguments['value']) * 2}\n",
        "test_arguments": {"value": 3},
    }
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(candidate)))]
    )
    monkeypatch.setattr(generator, "ask_llm", lambda messages: response)

    result = generator.generate_plugin_candidate(recipe())
    assert result["plugin_id"] == "double-value"
    assert result["test_arguments"] == {"value": 3}


def test_generate_plugin_candidate_rejects_wrong_capability(monkeypatch) -> None:
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=json.dumps({
                        "plugin_id": "x",
                        "manifest": {
                            "name": "x",
                            "version": "0.1.0",
                            "description": "x",
                            "parameters": {"type": "object"},
                            "capabilities": ["process"],
                        },
                        "source": "def run(arguments): return {'ok': True}",
                        "test_arguments": {},
                    })
                )
            )
        ]
    )
    monkeypatch.setattr(generator, "ask_llm", lambda messages: response)

    with pytest.raises(generator.PluginGenerationError):
        generator.generate_plugin_candidate(recipe())
