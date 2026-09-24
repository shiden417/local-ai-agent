from pathlib import Path

from agent.runtime import AgentRuntime
from agent.tools import create_default_tool_registry
from tools.replace_line import replace_line


def test_replace_line_changes_only_requested_line(tmp_path: Path) -> None:
    target = tmp_path / "notes.txt"
    target.write_text(
        "HEADER\nTOKEN\nMIDDLE\nTOKEN\nFOOTER\n",
        encoding="utf-8",
    )

    result = replace_line(
        tmp_path,
        {
            "path": "notes.txt",
            "line_number": 2,
            "new_text": "FIRST_TOKEN",
        },
    )

    assert result["ok"] is True
    assert target.read_text(encoding="utf-8") == (
        "HEADER\nFIRST_TOKEN\nMIDDLE\nTOKEN\nFOOTER\n"
    )


def test_replace_line_rejects_out_of_range(tmp_path: Path) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("A\n", encoding="utf-8")

    result = replace_line(
        tmp_path,
        {
            "path": "notes.txt",
            "line_number": 3,
            "new_text": "B",
        },
    )

    assert result["ok"] is False
    assert "outside the file" in result["error"]


def test_default_registry_contains_structured_edit_tools() -> None:
    names = set(create_default_tool_registry().names())
    assert "python_symbol_edit" in names
    assert "replace_line" in names


def test_runtime_exposes_python_structural_tool_for_python_mutation_tasks() -> None:
    runtime = AgentRuntime(Path("."))
    result = runtime._effective_task_requirements(
        "calculator に subtract(a, b) を追加してください。",
        "calculator に subtract(a, b) を追加してください。",
        is_follow_up=False,
    )

    assert result.file_mutation is True
