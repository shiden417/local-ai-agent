from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent.task import ProgressState


@dataclass(frozen=True)
class ProgressEvaluation:
    """Runtime-level judgment of whether a tool result advanced the task."""

    state: ProgressState
    reason: str


def evaluate_progress(
    tool_name: str,
    result: dict[str, Any],
    observation_is_new: bool,
) -> ProgressEvaluation:
    """Separate task progress from action identity and observation novelty.

    The evaluator stays intentionally small and deterministic. Tool-specific
    semantics are preferred where they are unambiguous; otherwise a new,
    successful observation is treated as progress.
    """

    if result.get("repeated_tool_call"):
        return ProgressEvaluation(
            ProgressState.BLOCKED,
            "The same action was already executed in this task.",
        )

    if result.get("user_rejected"):
        return ProgressEvaluation(
            ProgressState.BLOCKED,
            "The user rejected the requested operation.",
        )

    if not result.get("ok", False):
        return ProgressEvaluation(
            ProgressState.FAILED,
            "The tool execution failed.",
        )

    if tool_name == "search_memory" and not result.get("entries"):
        return ProgressEvaluation(
            ProgressState.NO_PROGRESS,
            "Memory search returned no entries.",
        )

    if not observation_is_new:
        return ProgressEvaluation(
            ProgressState.NO_PROGRESS,
            "The result adds no new observation.",
        )

    # Successful mutations/process actions are useful progress even when
    # their output is compact. Read/discovery tools count as progress when
    # they provide a genuinely new observation.
    return ProgressEvaluation(
        ProgressState.PROGRESSED,
        "The successful result added new task information or changed state.",
    )
