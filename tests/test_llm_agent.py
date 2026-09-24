import os

from agent import llm


def test_lm_studio_configuration_uses_default_or_environment_override() -> None:
    assert llm.LM_STUDIO_BASE_URL == "http://localhost:1234/v1"
    expected_model = os.getenv("LM_STUDIO_MODEL") or llm.DEFAULT_MODEL
    assert llm.MODEL == expected_model


def test_lm_studio_default_model_name() -> None:
    assert llm.DEFAULT_MODEL == "qwen/qwen3-8b"
