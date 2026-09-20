from agent.safety import requires_confirmation


def test_edit_file_requires_confirmation() -> None:
    assert requires_confirmation("edit_file", {}) is True


def test_read_only_command_does_not_require_confirmation() -> None:
    assert requires_confirmation(
        "execute_command",
        {"command": "git status"},
    ) is False


def test_destructive_command_requires_confirmation() -> None:
    assert requires_confirmation(
        "execute_command",
        {"command": "git reset --hard HEAD"},
    ) is True
