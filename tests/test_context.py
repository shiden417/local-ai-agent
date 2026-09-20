from agent.context import ContextManager


def test_context_manager_keeps_small_history_unchanged() -> None:
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "world"},
    ]

    manager = ContextManager(max_chars=10_000)

    assert manager.prepare(messages) == messages


def test_context_manager_compacts_old_history() -> None:
    large_output = "x" * 2_000
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "first request"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call-1",
                    "function": {"name": "read_file", "arguments": "{}"},
                }
            ],
            "content": "",
        },
        {
            "role": "tool",
            "name": "read_file",
            "content": (
                '{"ok": true, "path": "a.py", '
                f'"content": "{large_output}"}'
            ),
        },
        {"role": "assistant", "content": "first step"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call-2",
                    "function": {"name": "search_files", "arguments": "{}"},
                }
            ],
            "content": "",
        },
        {
            "role": "tool",
            "name": "search_files",
            "content": '{"ok": false, "error": "not found"}',
        },
        {"role": "user", "content": "second request"},
        {"role": "assistant", "content": "recent decision"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call-3",
                    "function": {"name": "read_file", "arguments": "{}"},
                }
            ],
            "content": "",
        },
        {
            "role": "tool",
            "name": "read_file",
            "content": '{"ok": true, "path": "b.py", "content": "recent"}',
        },
        {"role": "assistant", "content": "continue"},
    ]

    manager = ContextManager(
        max_chars=2_000,
        keep_recent_blocks=2,
    )

    compacted = manager.prepare(messages)

    assert any(
        message.get("role") == "system"
        and "Compacted history summary" in message.get("content", "")
        for message in compacted
    )
    assert any(
        message.get("content") == "continue"
        for message in compacted
    )

    tool_index = next(
        index
        for index, message in enumerate(compacted)
        if message.get("role") == "tool"
    )
    assert tool_index > 0
    assert compacted[tool_index - 1].get("role") == "assistant"


def test_context_summary_keeps_tool_status() -> None:
    messages = [
        {
            "role": "user",
            "content": "run it",
        },
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call-1",
                    "function": {
                        "name": "execute_command",
                        "arguments": "{}",
                    },
                }
            ],
            "content": "",
        },
        {
            "role": "tool",
            "name": "execute_command",
            "content": '{"ok": false, "exit_code": 1, "error": "test failed"}',
        },
    ]

    manager = ContextManager(max_chars=500)

    compacted = manager.prepare(messages)

    text = "\n".join(
        str(message.get("content", ""))
        for message in compacted
    )

    assert "execute_command" in text
    assert "exit_code=1" in text
    assert "test failed" in text
