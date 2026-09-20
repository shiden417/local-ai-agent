from __future__ import annotations

import json
import re
from typing import Any

from agent.llm import ask_llm
from agent.recipe_store import RecipeEntry

MAX_GENERATED_SOURCE_CHARS = 12_000

class PluginGenerationError(ValueError):
    """Raised when the local LLM cannot produce a valid Plugin candidate."""

def generate_plugin_candidate(recipe: RecipeEntry) -> dict[str, Any]:
    if not recipe.script.strip():
        raise PluginGenerationError("recipe script must not be empty")

    prompt = (
        "あなたはLocal AI AgentのCapability Generatorです。\n"
        "成功済みRecipeを再利用可能な小さなPlugin候補へ変換してください。\n\n"
        f"Recipe goal:\n{recipe.goal}\n\n"
        f"Recipe script:\n{recipe.script}\n\n"
        "返答はJSONオブジェクトだけにしてください。Markdown不要です。\n"
        "必須キー: plugin_id, manifest, source, test_arguments\n"
        "manifestにはname, version, description, parameters, capabilities, use_when, avoid_whenを含める。\n"
        "sourceはplugin.py全体で、top-level run(arguments)を定義する。\n"
        "test_argumentsはrun(arguments)が成功する最小の引数。\n"
        "capabilitiesはscript_executionのみ。versionは0.1.0。\n"
        "外部パッケージ、subprocess、ネットワークを使わない。Core/Runtime/Safety/Memory/Registryを変更しない。\n"
        "runはdictを返し、不正入力にはok=falseとerrorを返す。"
    )

    response = ask_llm([
        {
            "role": "system",
            "content": "Return only valid JSON. Generate a minimal deterministic Plugin candidate.",
        },
        {"role": "user", "content": prompt},
    ])
    message = response.choices[0].message
    content = getattr(message, "content", "") or ""
    candidate = _parse_json(content)
    _validate_candidate(candidate)
    return candidate

def _parse_json(content: str) -> dict[str, Any]:
    normalized = content.strip()
    if normalized.startswith("```"):
        normalized = re.sub(r"^```(?:json)?\s*", "", normalized, flags=re.IGNORECASE)
        normalized = re.sub(r"\s*```$", "", normalized)
    try:
        value = json.loads(normalized)
    except json.JSONDecodeError as exc:
        raise PluginGenerationError(f"generated Plugin response was not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise PluginGenerationError("generated Plugin response must be a JSON object")
    return value

def _validate_candidate(candidate: dict[str, Any]) -> None:
    required = ("plugin_id", "manifest", "source", "test_arguments")
    missing = [key for key in required if key not in candidate]
    if missing:
        raise PluginGenerationError(f"generated Plugin candidate missing: {', '.join(missing)}")
    manifest = candidate["manifest"]
    if not isinstance(manifest, dict):
        raise PluginGenerationError("generated manifest must be an object")
    source = candidate["source"]
    if not isinstance(source, str) or not source.strip():
        raise PluginGenerationError("generated Plugin source must not be empty")
    if len(source) > MAX_GENERATED_SOURCE_CHARS:
        raise PluginGenerationError(f"generated Plugin source exceeds {MAX_GENERATED_SOURCE_CHARS} characters")
    test_arguments = candidate["test_arguments"]
    if not isinstance(test_arguments, dict):
        raise PluginGenerationError("generated test_arguments must be an object")
    if manifest.get("capabilities") != ["script_execution"]:
        raise PluginGenerationError("generated Plugin must declare only script_execution capability")
    if "parameters" not in manifest or not isinstance(manifest["parameters"], dict):
        raise PluginGenerationError("generated manifest parameters must be a JSON Schema object")
