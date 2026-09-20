from litellm import completion

response = completion(
    model="ollama/qwen3:8b",
    messages=[
        {
            "role": "user",
            "content": "こんにちは。あなたは何ができますか？短く答えてください。"
        }
    ],
)

print(response.choices[0].message.content)