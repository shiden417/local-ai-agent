from pathlib import Path
from types import SimpleNamespace

from agent.trace import TraceRecorder, extract_usage


def test_trace_recorder_writes_structured_events(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    recorder = TraceRecorder(path, enabled=True)

    recorder.run_start(
        "run-1",
        task_id="task-1",
        mode="task",
        goal="調査",
        model="qwen/qwen3-8b",
    )
    recorder.run_end(
        "run-1",
        task_id="task-1",
        status="completed",
        iterations=2,
        tool_calls=1,
    )

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert '"event":"run_start"' in lines[0]
    assert '"task_id":"task-1"' in lines[0]
    assert '"event":"run_end"' in lines[1]


def test_extract_usage_reads_reasoning_tokens() -> None:
    response = SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=12,
            completion_tokens=8,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=3),
        )
    )

    assert extract_usage(response) == {
        "prompt_tokens": 12,
        "completion_tokens": 8,
        "reasoning_tokens": 3,
    }
