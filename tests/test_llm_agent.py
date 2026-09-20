from agent import llm


def test_lm_studio_defaults() -> None:
    assert llm.LM_STUDIO_BASE_URL == "http://localhost:1234/v1"
    assert llm.MODEL == "qwen/qwen3-8b"
