from litellm import completion

MODEL = "ollama/qwen3:8b"


def ask_llm(messages: list[dict[str, str]]) -> str:
    """Send the conversation to the local Ollama model through LiteLLM."""
    response = completion(
        model=MODEL,
        messages=messages,
    )
    return response.choices[0].message.content or ""
