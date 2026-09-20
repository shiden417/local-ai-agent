from pathlib import Path

from agent.safety import requires_confirmation
from agent.tool_registry import ToolDefinition, ToolRegistry


def test_registered_mutating_tool_requires_confirmation(tmp_path: Path) -> None:
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="mutate",
            description="Mutate local state",
            parameters={"type": "object", "properties": {}, "required": []},
            handler=lambda _working_directory, _arguments: {"ok": True},
            requires_confirmation=True,
        )
    )

    assert requires_confirmation("mutate", {}, registry) is True


def test_read_only_command_does_not_require_confirmation() -> None:
    registry = ToolRegistry()

    assert requires_confirmation(
        "execute_command",
        {"command": "git status"},
        registry,
    ) is False


def test_destructive_command_requires_confirmation() -> None:
    registry = ToolRegistry()

    assert requires_confirmation(
        "execute_command",
        {"command": "git reset --hard HEAD"},
        registry,
    ) is True


def test_power_shell_write_command_requires_confirmation() -> None:
    registry = ToolRegistry()

    assert requires_confirmation(
        "execute_command",
        {"command": "Set-Content -Path example.txt -Value hello"},
        registry,
    ) is True


def test_output_redirection_requires_confirmation() -> None:
    registry = ToolRegistry()

    assert requires_confirmation(
        "execute_command",
        {"command": "Get-Date > example.txt"},
        registry,
    ) is True
