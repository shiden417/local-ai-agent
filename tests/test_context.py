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
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "first request"},
        {"role": "tool", "name": "read_file", "content": '{"ok": true, "path": "a.py", "content": "x" * 1000}'},
        {"role": "assistant", "content": "first step"},
        {"role": "tool", "name": "search_files", "content": '{"ok": false, "error": "not found"}'},
        {"role": "user", "content": "second request"},
        {"role": "assistant", "content": "recent decision"},
        {"role": "tool", "name": "read_file", "content": '{"ok": true, "path": "b.py", "content": "recent"}'},
        {"role": "assistant", "content": "continue"},
    ]

    manager = ContextManager(
        max_chars=1_500,
        keep_recent_messages=4,
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


def test_context_summary_keeps_tool_status() -> None:
    messages = [
        {"role": "tool", "name": "execute_command", "content": '{"ok": false, "exit_code": 1, "error": "test failed"}'},
    ]

    manager = ContextManager(max_chars=10)

    compacted = manager.prepare(messages)

    assert "execute_command" in compacted[0]["content"]
    assert "exit_code=1" in compacted[0]["content"]
    assert "test failed" in compacted[0]["content"]
