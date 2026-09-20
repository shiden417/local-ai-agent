from agent.observation import truncate_text


def test_truncate_text_keeps_short_text() -> None:
    text = "short"

    result, truncated = truncate_text(text, max_chars=10)

    assert result == text
    assert truncated is False


def test_truncate_text_preserves_both_ends() -> None:
    text = "abcdefghij"

    result, truncated = truncate_text(text, max_chars=7)

    assert truncated is True
    assert "..." in result
    assert result.startswith("a")
    assert result.endswith("j")
