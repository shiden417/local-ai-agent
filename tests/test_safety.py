from pathlib import Path

from agent.safety import AUTO_ALLOW, AUTO_ASK, AUTO_DENY, SafetyPolicy, validate_command_scope
from agent.tool_registry import ToolDefinition, ToolRegistry


def test_registered_mutating_tool_requires_approval_decision(tmp_path: Path) -> None:
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

    policy = SafetyPolicy(tmp_path / "approvals.json")
    assert policy.decide("mutate", {}, registry, tmp_path) == AUTO_ASK


def test_read_only_command_is_allowed(tmp_path: Path) -> None:
    policy = SafetyPolicy(tmp_path / "approvals.json")
    registry = ToolRegistry()

    assert policy.decide(
        "execute_command",
        {"command": "git status"},
        registry,
        tmp_path,
    ) == AUTO_ALLOW


def test_destructive_command_requires_approval(tmp_path: Path) -> None:
    policy = SafetyPolicy(tmp_path / "approvals.json")
    registry = ToolRegistry()

    assert policy.decide(
        "execute_command",
        {"command": "git reset --hard HEAD"},
        registry,
        tmp_path,
    ) == AUTO_ASK


def test_power_shell_write_command_requires_approval(tmp_path: Path) -> None:
    policy = SafetyPolicy(tmp_path / "approvals.json")
    registry = ToolRegistry()

    assert policy.decide(
        "execute_command",
        {"command": "Set-Content -Path example.txt -Value hello"},
        registry,
        tmp_path,
    ) == AUTO_ASK


def test_output_redirection_requires_approval(tmp_path: Path) -> None:
    policy = SafetyPolicy(tmp_path / "approvals.json")
    registry = ToolRegistry()

    assert policy.decide(
        "execute_command",
        {"command": "Get-Date > example.txt"},
        registry,
        tmp_path,
    ) == AUTO_ASK


def test_absolute_external_command_path_requires_approval_or_denial(tmp_path: Path) -> None:
    policy = SafetyPolicy(tmp_path / "approvals.json")
    registry = ToolRegistry()

    assert policy.decide(
        "execute_command",
        {"command": "Get-ChildItem C:\\Windows\\System32"},
        registry,
        tmp_path,
    ) == AUTO_DENY


def test_hard_deny_command_is_blocked_even_without_confirmation_metadata(
    tmp_path: Path,
) -> None:
    policy = SafetyPolicy(tmp_path / "approvals.json")
    registry = ToolRegistry()

    assert policy.decide(
        "execute_command",
        {"command": "shutdown.exe /s /t 0"},
        registry,
        tmp_path,
    ) == AUTO_DENY


def test_workspace_absolute_path_does_not_fail_scope_validation(
    tmp_path: Path,
) -> None:
    command = f"Get-ChildItem '{tmp_path.resolve()}'"

    assert validate_command_scope(command, tmp_path) is None


def test_external_absolute_path_is_blocked(
    tmp_path: Path,
) -> None:
    policy = SafetyPolicy(tmp_path / "approvals.json")
    registry = ToolRegistry()

    assert (
        policy.decide(
            "execute_command",
            {"command": "Get-ChildItem C:\\Windows\\System32"},
            registry,
            tmp_path,
        )
        == AUTO_DENY
    )


def test_parent_directory_traversal_is_blocked(
    tmp_path: Path,
) -> None:
    policy = SafetyPolicy(tmp_path / "approvals.json")
    registry = ToolRegistry()

    assert (
        policy.decide(
            "execute_command",
            {"command": "Get-ChildItem ..\\outside"},
            registry,
            tmp_path,
        )
        == AUTO_DENY
    )


def test_external_local_path_requires_approval_for_mutating_tool(
    tmp_path: Path,
) -> None:
    registry = ToolRegistry()
    policy = SafetyPolicy(tmp_path / "approvals.json")

    assert (
        policy.decide(
            "file_mutation",
            {
                "operation": "edit",
                "path": r"C:\Users\example\OtherProject\test.txt",
                "search_text": "old",
                "replace_text": "new",
            },
            registry,
            tmp_path,
        )
        == AUTO_ASK
    )


def test_workspace_absolute_path_is_allowed_for_read_tool(
    tmp_path: Path,
) -> None:
    registry = ToolRegistry()
    policy = SafetyPolicy(tmp_path / "approvals.json")

    assert (
        policy.decide(
            "list_directory",
            {"path": str(tmp_path.resolve())},
            registry,
            tmp_path,
        )
        == AUTO_ALLOW
    )


def test_learned_approval_changes_ask_to_allow(tmp_path: Path) -> None:
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
    policy = SafetyPolicy(tmp_path / "approvals.json")
    key = policy.approval_key("mutate", {}, tmp_path)
    policy.allow(key, "approved")

    assert policy.decide("mutate", {}, registry, tmp_path) == AUTO_ALLOW
