from agent.conversation import ConversationManager


def test_conversation_manager_keeps_recent_turns() -> None:
    manager = ConversationManager(max_turns=2)

    manager.add_turn("one", "first")
    manager.add_turn("two", "second")
    manager.add_turn("three", "third")

    assert manager.recent_messages() == [
        {"role": "user", "content": "two"},
        {"role": "assistant", "content": "second"},
        {"role": "user", "content": "three"},
        {"role": "assistant", "content": "third"},
    ]


def test_conversation_manager_respects_character_limit() -> None:
    manager = ConversationManager(max_turns=8, max_chars=30)

    manager.add_turn("old", "a" * 20)
    manager.add_turn("new", "b" * 20)

    messages = manager.recent_messages()

    assert messages == [
        {"role": "user", "content": "new"},
        {"role": "assistant", "content": "b" * 20},
    ]


def test_conversation_manager_clear() -> None:
    manager = ConversationManager()
    manager.add_turn("hello", "hi")

    manager.clear()

    assert manager.messages == []
