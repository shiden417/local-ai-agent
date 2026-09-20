from agent import llm


def test_lm_studio_defaults() -> None:
    assert llm.LM_STUDIO_BASE_URL.startswith("http://localhost:")
    assert llm.MODEL == "qwen/qwen3-8b"
