from pathlib import Path

from agent.command_policy import AUTO_ALLOW, AUTO_ASK, AUTO_DENY, classify_auto_mode
from agent.tools import create_default_tool_registry


def test_auto_mode_allows_workspace_file_edit() -> None:
    registry = create_default_tool_registry()
    workspace = Path.cwd()

    decision = classify_auto_mode(
        "file_mutation",
        {"operation": "edit", "path": "README.md"},
        registry,
        workspace,
    )

    assert decision == AUTO_ALLOW


def test_auto_mode_asks_for_delete_and_python_script() -> None:
    registry = create_default_tool_registry()
    workspace = Path.cwd()

    assert classify_auto_mode(
        "file_mutation",
        {"operation": "delete", "path": "README.md"},
        registry,
        workspace,
    ) == AUTO_ASK
    assert classify_auto_mode(
        "run_python_script",
        {"script": "print('ok')"},
        registry,
        workspace,
    ) == AUTO_ASK


def test_auto_mode_allows_normal_command_but_asks_for_network() -> None:
    registry = create_default_tool_registry()
    workspace = Path.cwd()

    assert classify_auto_mode(
        "execute_command",
        {"command": "python -m pytest -q"},
        registry,
        workspace,
    ) == AUTO_ALLOW
    assert classify_auto_mode(
        "execute_command",
        {"command": "curl https://example.com"},
        registry,
        workspace,
    ) == AUTO_ASK


def test_auto_mode_denies_force_push_and_shutdown() -> None:
    registry = create_default_tool_registry()
    workspace = Path.cwd()

    assert classify_auto_mode(
        "execute_command",
        {"command": "git push --force origin main"},
        registry,
        workspace,
    ) == AUTO_DENY
    assert classify_auto_mode(
        "execute_command",
        {"command": "shutdown.exe /s /t 0"},
        registry,
        workspace,
    ) == AUTO_DENY
