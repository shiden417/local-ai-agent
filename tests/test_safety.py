from pathlib import Path

from agent.safety import requires_confirmation, validate_command_scope
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


def test_absolute_external_command_path_requires_confirmation() -> None:
    registry = ToolRegistry()

    assert requires_confirmation(
        "execute_command",
        {"command": "Get-ChildItem C:\\Windows\\System32"},
        registry,
    ) is True


def test_workspace_absolute_path_does_not_fail_scope_validation(
    tmp_path: Path,
) -> None:
    command = f"Get-ChildItem '{tmp_path.resolve()}'"

    assert validate_command_scope(command, tmp_path) is None


def test_external_absolute_path_is_blocked(
    tmp_path: Path,
) -> None:
    assert (
        validate_command_scope(
            "Get-ChildItem C:\\Windows\\System32",
            tmp_path,
        )
        is not None
    )


def test_parent_directory_traversal_is_blocked(
    tmp_path: Path,
) -> None:
    assert (
        validate_command_scope(
            "Get-ChildItem ..\\outside",
            tmp_path,
        )
        is not None
    )


def test_workspace_absolute_windows_path_with_spaces_is_allowed(
    tmp_path: Path,
) -> None:
    workspace = str(tmp_path.resolve())
    command = f'Get-ChildItem "{workspace}"'

    assert validate_command_scope(command, tmp_path) is None


def test_external_local_path_is_allowed_for_read_tool_without_confirmation(
    tmp_path: Path,
) -> None:
    registry = ToolRegistry()

    assert (
        requires_confirmation(
            "list_directory",
            {"path": r"C:\Users\example\OtherProject"},
            registry,
            tmp_path,
        )
        is False
    )


def test_external_local_path_requires_confirmation_for_mutating_tool(
    tmp_path: Path,
) -> None:
    registry = ToolRegistry()

    assert (
        requires_confirmation(
            "edit_file",
            {
                "path": r"C:\Users\example\OtherProject\test.txt",
                "search_text": "old",
                "replace_text": "new",
            },
            registry,
            tmp_path,
        )
        is True
    )


def test_workspace_absolute_path_does_not_require_extra_confirmation_for_read_tool(
    tmp_path: Path,
) -> None:
    registry = ToolRegistry()

    assert (
        requires_confirmation(
            "list_directory",
            {"path": str(tmp_path.resolve())},
            registry,
            tmp_path,
        )
        is False
    )


def test_delete_file_requires_confirmation_for_external_path(tmp_path: Path) -> None:
    registry = ToolRegistry()

    assert requires_confirmation(
        "delete_file",
        {"path": r"C:\Users\example\OtherProject\test.txt"},
        registry,
        tmp_path,
    ) is True
