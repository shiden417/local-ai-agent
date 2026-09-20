from agent.session import SessionManager


def test_session_manager_keeps_small_task_history_unchanged() -> None:
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "world"},
    ]
    manager = SessionManager(max_task_chars=10_000)
    assert manager.prepare_task_messages(messages) == messages


def test_session_manager_compacts_old_task_history() -> None:
    large_output = "x" * 2_000
    large_output_json = '{"ok": true, "path": "a.py", "content": "' + large_output + '"}'
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "first request"},
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "call-1", "function": {"name": "read_file", "arguments": "{}"}}
            ],
            "content": "",
        },
        {"role": "tool", "name": "read_file", "content": large_output_json},
        {"role": "assistant", "content": "first step"},
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "call-2", "function": {"name": "search_files", "arguments": "{}"}}
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
                {"id": "call-3", "function": {"name": "read_file", "arguments": "{}"}}
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

    manager = SessionManager(max_task_chars=2_000, keep_recent_blocks=3)
    compacted = manager.prepare_task_messages(messages)

    assert any(
        message.get("role") == "system"
        and "Compacted history summary" in message.get("content", "")
        for message in compacted
    )
    assert any(message.get("content") == "continue" for message in compacted)

    tool_index = next(
        index for index, message in enumerate(compacted) if message.get("role") == "tool"
    )
    assert tool_index > 0
    assert compacted[tool_index - 1].get("role") == "assistant"


def test_session_manager_summary_keeps_tool_status() -> None:
    messages = [
        {"role": "user", "content": "run it"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call-1",
                    "function": {"name": "execute_command", "arguments": "{}"},
                }
            ],
            "content": "",
        },
        {
            "role": "tool",
            "name": "execute_command",
            "content": '{"ok": false, "exit_code": 1, "error": "test failed"}',
        },
        {"role": "user", "content": "continue"},
    ]

    manager = SessionManager(max_task_chars=500, keep_recent_blocks=1)
    compacted = manager.prepare_task_messages(messages)
    text = "\n".join(str(message.get("content", "")) for message in compacted)

    assert "execute_command" in text
    assert "exit_code=1" in text
    assert "test failed" in text


def test_session_manager_keeps_recent_conversation_turns() -> None:
    manager = SessionManager(max_conversation_turns=2)
    manager.add_conversation_turn("one", "first")
    manager.add_conversation_turn("two", "second")
    manager.add_conversation_turn("three", "third")

    assert manager.recent_conversation_messages() == [
        {"role": "user", "content": "two"},
        {"role": "assistant", "content": "second"},
        {"role": "user", "content": "three"},
        {"role": "assistant", "content": "third"},
    ]


def test_session_manager_respects_conversation_character_limit() -> None:
    manager = SessionManager(max_conversation_turns=8, max_conversation_chars=30)
    manager.add_conversation_turn("old", "a" * 20)
    manager.add_conversation_turn("new", "b" * 20)

    assert manager.recent_conversation_messages() == [
        {"role": "user", "content": "new"},
        {"role": "assistant", "content": "b" * 20},
    ]


def test_session_manager_carries_previous_task_context() -> None:
    manager = SessionManager()
    manager.remember_task(
        "LM Studioの移行状況を調べる",
        "移行済みです",
        [
            {
                "role": "tool",
                "name": "search_web",
                "content": '{"ok": true, "results": [{"title": "LM Studio", "snippet": "docs", "url": "https://example.com"}]}',
            }
        ],
    )

    assert manager.has_context is True
    assert manager.last_goal == "LM Studioの移行状況を調べる"
    prompt = manager.prompt_block()
    assert "LM Studio" in prompt
    assert "https://example.com" in prompt


def test_session_manager_clear_removes_all_context() -> None:
    manager = SessionManager()
    manager.add_conversation_turn("hello", "hi")
    manager.remember_task("task", "done", [])

    manager.clear()

    assert manager.has_context is False
    assert manager.recent_conversation_messages() == []
