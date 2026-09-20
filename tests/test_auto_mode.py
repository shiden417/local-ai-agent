from pathlib import Path

from agent.safety import AUTO_ALLOW, AUTO_ASK, AUTO_DENY, SafetyPolicy
from agent.tools import create_default_tool_registry


def test_auto_mode_allows_workspace_file_edit(tmp_path: Path) -> None:
    registry = create_default_tool_registry()
    workspace = Path.cwd()

    decision = SafetyPolicy(Path(tmp_path) / "approvals.json").decide(
        "file_mutation",
        {"operation": "edit", "path": "README.md"},
        registry,
        workspace,
    )

    assert decision == AUTO_ALLOW


def test_auto_mode_asks_for_delete_and_python_script(tmp_path: Path) -> None:
    registry = create_default_tool_registry()
    workspace = Path.cwd()

    assert SafetyPolicy(Path(tmp_path) / "approvals.json").decide(
        "file_mutation",
        {"operation": "delete", "path": "README.md"},
        registry,
        workspace,
    ) == AUTO_ASK
    assert SafetyPolicy(Path(tmp_path) / "approvals.json").decide(
        "run_python_script",
        {"script": "print('ok')"},
        registry,
        workspace,
    ) == AUTO_ASK


def test_auto_mode_allows_normal_command_but_asks_for_network(tmp_path: Path) -> None:
    registry = create_default_tool_registry()
    workspace = Path.cwd()

    assert SafetyPolicy(Path(tmp_path) / "approvals.json").decide(
        "execute_command",
        {"command": "python -m pytest -q"},
        registry,
        workspace,
    ) == AUTO_ALLOW
    assert SafetyPolicy(Path(tmp_path) / "approvals.json").decide(
        "execute_command",
        {"command": "curl https://example.com"},
        registry,
        workspace,
    ) == AUTO_ASK


def test_auto_mode_denies_force_push_and_shutdown(tmp_path: Path) -> None:
    registry = create_default_tool_registry()
    workspace = Path.cwd()

    assert SafetyPolicy(Path(tmp_path) / "approvals.json").decide(
        "execute_command",
        {"command": "git push --force origin main"},
        registry,
        workspace,
    ) == AUTO_DENY
    assert SafetyPolicy(Path(tmp_path) / "approvals.json").decide(
        "execute_command",
        {"command": "shutdown.exe /s /t 0"},
        registry,
        workspace,
    ) == AUTO_DENY
