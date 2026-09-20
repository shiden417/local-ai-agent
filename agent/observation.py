def truncate_text(text: str, max_chars: int = 8_000) -> tuple[str, bool]:
    """Keep tool observations bounded while preserving both ends."""
    if len(text) <= max_chars:
        return text, False

    marker = "\n... [output truncated] ...\n"
    available = max_chars - len(marker)
    head = max(1, available // 2)
    tail = max(1, available - head)
    return text[:head] + marker + text[-tail:], True
