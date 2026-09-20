from pathlib import Path

from agent.approval import ApprovalPolicy, approval_key


def test_approval_policy_persists_explicit_allow(tmp_path: Path) -> None:
    path = tmp_path / "approvals.json"
    first = ApprovalPolicy(path)
    first.allow("file_mutation:edit:workspace", "edit local files")

    second = ApprovalPolicy(path)
    assert second.is_allowed("file_mutation:edit:workspace") is True
    assert second.is_allowed("file_mutation:delete:abc") is False
    assert second.entries()[0]["description"] == "edit local files"


def test_approval_policy_clear_removes_learned_rules(tmp_path: Path) -> None:
    policy = ApprovalPolicy(tmp_path / "approvals.json")
    policy.allow("tool:key")
    policy.clear()

    assert policy.entries() == []


def test_approval_key_uses_different_granularity_by_risk(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    edit_a = approval_key(
        "file_mutation",
        {"operation": "edit", "path": "a.txt", "replace_text": "A"},
        workspace,
    )
    edit_b = approval_key(
        "file_mutation",
        {"operation": "edit", "path": "b.txt", "replace_text": "B"},
        workspace,
    )
    delete_a = approval_key(
        "file_mutation",
        {"operation": "delete", "path": "a.txt"},
        workspace,
    )

    assert edit_a == edit_b
    assert delete_a != edit_a
