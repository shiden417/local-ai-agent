from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run deterministic live JARVIS tasks against LM Studio."
    )
    parser.add_argument(
        "--model",
        help="LM Studio model ID. Defaults to LM_STUDIO_MODEL or the project default.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10,
        help="Maximum Agent iterations per task.",
    )
    parser.add_argument(
        "--thinking-mode",
        choices=("default", "think", "no_think"),
        help="Optional Qwen3 Thinking mode for this benchmark run.",
    )
    parser.add_argument(
        "--keep-workspace",
        action="store_true",
        help="Keep the temporary benchmark workspace for inspection.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON file for benchmark results.",
    )
    return parser.parse_args()


def _seed_workspace(root: Path) -> None:
    (root / "calculator.py").write_text(
        "def add(a, b):\n"
        "    return a - b\n\n"
        "def multiply(a, b):\n"
        "    return a * b\n",
        encoding="utf-8",
    )
    (root / "test_calculator.py").write_text(
        "from calculator import add, multiply\n\n"
        "def test_add():\n"
        "    assert add(2, 3) == 5\n\n"
        "def test_multiply():\n"
        "    assert multiply(2, 3) == 6\n",
        encoding="utf-8",
    )


def _run_task(runtime, prompt: str) -> tuple[str, float, dict[str, int]]:
    started = time.perf_counter()
    before = runtime.trace.summary()
    result = runtime.run(prompt)
    elapsed = time.perf_counter() - started
    after = runtime.trace.summary()
    delta = {key: int(after.get(key, 0)) - int(before.get(key, 0)) for key in after}
    return result, elapsed, delta


def _task_used_tool(runtime, names: set[str]) -> bool:
    task = runtime.task
    if task is None:
        return False
    return any(observation.tool in names and observation.ok for observation in task.observations)

def _check_exact(path: Path, expected: str) -> bool:
    try:
        return path.read_text(encoding="utf-8") == expected
    except OSError:
        return False


def _trace_summary(path: Path) -> dict[str, int]:
    metrics = {
        "llm_calls": 0,
        "tool_calls": 0,
        "llm_duration_ms": 0,
        "tool_duration_ms": 0,
        "prompt_chars": 0,
        "tool_schema_chars": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "reasoning_tokens": 0,
    }
    if not path.exists():
        return metrics
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("event") == "llm":
            metrics["llm_calls"] += 1
            metrics["llm_duration_ms"] += int(event.get("duration_ms", 0) or 0)
            metrics["prompt_chars"] += int(event.get("prompt_chars", 0) or 0)
            metrics["tool_schema_chars"] += int(event.get("tool_schema_chars", 0) or 0)
            metrics["prompt_tokens"] += int(event.get("prompt_tokens", 0) or 0)
            metrics["completion_tokens"] += int(event.get("completion_tokens", 0) or 0)
            metrics["reasoning_tokens"] += int(event.get("reasoning_tokens", 0) or 0)
        elif event.get("event") == "tool":
            metrics["tool_calls"] += 1
            metrics["tool_duration_ms"] += int(event.get("duration_ms", 0) or 0)
    return metrics

def run_benchmark(root: Path, *, model: str | None, max_iterations: int, output: Path | None = None, thinking_mode: str | None = None) -> int:
    if model:
        os.environ["LM_STUDIO_MODEL"] = model
    if thinking_mode:
        os.environ["LM_STUDIO_THINKING_MODE"] = thinking_mode

    from agent.runtime import AgentRuntime
    from agent.llm import MODEL
    from agent.trace import TraceRecorder
    from agent.memory import MemoryStore
    from agent.tools import create_default_tool_registry

    _seed_workspace(root)
    trace_path = root / "trace.jsonl"
    runtime = AgentRuntime(
        working_directory=root,
        max_iterations=max_iterations,
        tool_registry=create_default_tool_registry(
            memory_store=MemoryStore(root / "memory.json")
        ),
        confirm=lambda _message: True,
        trace_recorder=TraceRecorder(trace_path),
    )

    print("J.A.R.V.I.S. Agent Benchmark")
    print(f"Model: {MODEL}")
    if thinking_mode:
        print(f"Thinking mode: {thinking_mode}")
    print(f"Workspace: {root}")
    print()

    tasks = [
        (
            "Task 1: file creation",
            "このworkspaceに hello.txt を新規作成してください。内容は1行だけで Hello JARVIS としてください。作成したことを確認してください。",
            lambda: _check_exact(root / "hello.txt", "Hello JARVIS") and _task_used_tool(runtime, {"file_mutation"}),
        ),
        (
            "Task 2: session follow-up",
            "そのファイルの2行目に Session Context works を追加してください。既存の1行目は変更しないでください。確認してください。",
            lambda: _check_exact(root / "hello.txt", "Hello JARVIS\nSession Context works") and _task_used_tool(runtime, {"file_mutation"}),
        ),
        (
            "Task 3: read-only investigation",
            "calculator.py の add 関数を調査して、現在の実装内容を確認してください。ファイルは変更しないでください。",
            lambda: _check_exact(root / "calculator.py", "def add(a, b):\n    return a - b\n\ndef multiply(a, b):\n    return a * b\n") and _task_used_tool(runtime, {"read_file", "search_files"}),
        ),
        (
            "Task 4: file search",
            "workspace内で multiply という語があるファイルを検索して確認してください。ファイルを変更しないでください。",
            lambda: _task_used_tool(runtime, {"search_files"}),
        ),
        (
            "Task 5: investigate, edit, test",
            "calculator.py を調査してください。add関数にバグがあります。原因を修正し、python -m pytest -q を実行して、全テストが成功することを確認してください。test_calculator.py は変更しないでください。",
            lambda: (_check_exact(root / "calculator.py", "def add(a, b):\n    return a + b\n\ndef multiply(a, b):\n    return a * b\n") and _check_exact(root / "test_calculator.py", "from calculator import add, multiply\n\ndef test_add():\n    assert add(2, 3) == 5\n\ndef test_multiply():\n    assert multiply(2, 3) == 6\n") and _task_used_tool(runtime, {"file_mutation"}) and _task_used_tool(runtime, {"execute_command"})),
        ),
        (
            "Task 6: process execution",
            "このworkspaceで python -c を使って JARVIS benchmark と表示するコマンドを実行し、終了コード0を確認してください。",
            lambda: _task_used_tool(runtime, {"execute_command"}),
        ),
        (
            "Task 7: file deletion",
            "hello.txt を削除してください。削除されたことを確認してください。",
            lambda: (not (root / "hello.txt").exists()) and _task_used_tool(runtime, {"file_mutation"}),
        ),
        (
            "Task 8: memory",
            "このBenchmarkの識別子 jarvis-benchmark をMemoryに保存し、その後検索して保存できたことを確認してください。",
            lambda: (root / "memory.json").exists() and _task_used_tool(runtime, {"save_memory", "search_memory"}),
        ),
    ]

    results: list[dict[str, object]] = []
    started_all = time.perf_counter()
    passed = 0
    for label, prompt, check in tasks:
        print(f"[RUN] {label}")
        try:
            result, elapsed, task_metrics = _run_task(runtime, prompt)
            ok = check()
        except Exception as exc:
            result = f"{type(exc).__name__}: {exc}"
            elapsed = 0.0
            task_metrics = {}
            ok = False

        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {label} ({elapsed:.1f}s)")
        print(f"Final: {result}")
        print()
        if ok:
            passed += 1
        results.append(
            {
                "label": label,
                "passed": ok,
                "elapsed_seconds": round(elapsed, 3),
                "final": str(result),
                "metrics": task_metrics,
            }
        )

    metrics = _trace_summary(trace_path)
    report = {
        "model": MODEL,
        "max_iterations": max_iterations,
        "total_elapsed_seconds": round(time.perf_counter() - started_all, 3),
        "passed": passed,
        "total_tasks": len(tasks),
        "tasks": results,
        "metrics": metrics,
    }
    if output is not None:
        output = output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"JSON report: {output}")

    print(
        f"Metrics: LLM={metrics['llm_calls']} calls, "
        f"Tool={metrics['tool_calls']} calls, "
        f"LLM={metrics['llm_duration_ms']}ms, "
        f"Reasoning={metrics['reasoning_tokens']} tokens"
    )
    print(f"Result: {passed}/{len(tasks)} tasks passed")
    return 0 if passed == len(tasks) else 1


def main() -> int:
    args = _parse_args()
    if args.max_iterations < 1:
        raise SystemExit("--max-iterations must be at least 1")

    if args.keep_workspace:
        workspace = Path(tempfile.mkdtemp(prefix="jarvis-benchmark-"))
        print(f"Benchmark workspace: {workspace}")
        return run_benchmark(
            workspace,
            model=args.model,
            max_iterations=args.max_iterations,
            output=args.output,
            thinking_mode=args.thinking_mode,
        )

    with tempfile.TemporaryDirectory(prefix="jarvis-benchmark-") as temp_dir:
        return run_benchmark(
            Path(temp_dir),
            model=args.model,
            max_iterations=args.max_iterations,
            output=args.output,
            thinking_mode=args.thinking_mode,
        )


if __name__ == "__main__":
    raise SystemExit(main())
