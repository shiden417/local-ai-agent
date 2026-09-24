import importlib
import os

from agent import llm


def test_lm_studio_defaults(monkeypatch) -> None:
    monkeypatch.delenv("LM_STUDIO_MODEL", raising=False)
    reloaded = importlib.reload(llm)
    try:
        assert reloaded.LM_STUDIO_BASE_URL == "http://localhost:1234/v1"
        assert reloaded.MODEL == reloaded.DEFAULT_MODEL
    finally:
        if os.environ.get("LM_STUDIO_MODEL"):
            importlib.reload(llm)
        else:
            importlib.reload(llm)


def test_lm_studio_model_can_be_overridden(monkeypatch) -> None:
    monkeypatch.setenv("LM_STUDIO_MODEL", "google/gemma-4-e4b")
    reloaded = importlib.reload(llm)
    try:
        assert reloaded.MODEL == "google/gemma-4-e4b"
    finally:
        importlib.reload(llm)
