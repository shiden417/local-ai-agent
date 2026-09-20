from __future__ import annotations

from typing import Any


STATUS_SUCCESS = "success"
STATUS_BLOCKED = "blocked"
STATUS_NOT_FOUND = "not_found"
STATUS_INVALID_INPUT = "invalid_input"
STATUS_PERMISSION_DENIED = "permission_denied"
STATUS_USER_REJECTED = "user_rejected"
STATUS_TIMEOUT = "timeout"
STATUS_EXTERNAL_FAILURE = "external_failure"
STATUS_FAILED = "failed"


def classify_tool_outcome(tool_name: str, result: dict[str, Any]) -> str:
    if bool(result.get("user_rejected")):
        return STATUS_USER_REJECTED
    if bool(result.get("blocked")) or bool(result.get("recovery_blocked")):
        return STATUS_BLOCKED
    if bool(result.get("timed_out")):
        return STATUS_TIMEOUT
    if bool(result.get("ok")):
        return STATUS_SUCCESS

    error = " ".join(
        str(result.get(key, ""))
        for key in ("error", "stderr", "message")
    ).lower()

    if "does not exist" in error or "not found" in error or "file not exist" in error:
        return STATUS_NOT_FOUND
    if "permission" in error or "access denied" in error or "forbidden" in error:
        return STATUS_PERMISSION_DENIED
    if "must be" in error or "invalid" in error or "required" in error:
        return STATUS_INVALID_INPUT
    if tool_name == "search_web":
        return STATUS_EXTERNAL_FAILURE

    return STATUS_FAILED


def recovery_guidance(tool_name: str, status: str) -> str:
    guidance = {
        STATUS_NOT_FOUND: (
            "Recovery Guide: The previous target was not found. "
            "Do not guess another path repeatedly. "
            "Use search_files or list_directory to discover the correct target first."
        ),
        STATUS_INVALID_INPUT: (
            "Recovery Guide: The previous Tool arguments were invalid. "
            "Correct the arguments from the concrete error message before retrying."
        ),
        STATUS_PERMISSION_DENIED: (
            "Recovery Guide: The previous action was blocked by permissions. "
            "Do not bypass the policy. Use an allowed read-only alternative or ask the user when a decision is required."
        ),
        STATUS_USER_REJECTED: (
            "Recovery Guide: The user rejected the previous action. "
            "Do not repeat it. Explain the constraint or choose a safe alternative."
        ),
        STATUS_TIMEOUT: (
            "Recovery Guide: The previous operation timed out. "
            "Do not repeat the identical long-running action. Narrow the scope or choose a smaller diagnostic step."
        ),
        STATUS_EXTERNAL_FAILURE: (
            "Recovery Guide: The external information lookup failed. "
            "Do not repeat the identical query more than once. Try one directly relevant alternate query, then reassess."
        ),
        STATUS_BLOCKED: (
            "Recovery Guide: The previous Tool was blocked by Runtime safety policy. "
            "Do not attempt to bypass the policy. Choose a policy-compliant alternative."
        ),
        STATUS_FAILED: (
            "Recovery Guide: The previous Tool failed. "
            "Use the error details to choose one directly relevant alternative; do not retry the same failing action unchanged."
        ),
    }
    return guidance.get(
        status,
        f"Recovery Guide: Reassess the failed Tool '{tool_name}' using the latest error details.",
    )
