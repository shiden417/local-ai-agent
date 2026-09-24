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
        description="Run failure-recovery JARVIS tasks against LM Studio."
    )
    parser.add_argument("--model")
    parser.add_argument("--max-iterations", type=int, default=10)
    parser.add_argument(
        "--thinking-mode",
        choices=("default", "think", "no_think"),
    )
    parser.add_argument("--keep-workspace", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _seed_workspace(root: Path) -> None:
    (root / "recovery.txt").write_text("OLD_VALUE\n", encoding="utf-8")
    (root / "calculator.py").write_text(
        "def add(a, b):\n"
        "    return a + 1\n\n"
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
    (root / "notes.txt").write_text(
        "alpha\nneedle\nomega\n",
        encoding="utf-8",
    )


def _run_task(runtime, prompt: str) -> tuple[str, float, dict[str, int]]:
    started = time.perf_counter()
    before = runtime.trace.summary()
    result = runtime.run(prompt)
    elapsed = time.perf_counter() - started
    after = runtime.trace.summary()
    delta = {
        key: int(after.get(key, 0)) - int(before.get(key, 0))
        for key in after
    }
    return result, elapsed, delta


def _tool_results(runtime, names: set[str]) -> list[dict[str, object]]:
    task = runtime.current_task
    if task is None:
        return []
    results: list[dict[str, object]] = []
    for message in task.messages:
        if message.get("role") != "tool" or message.get("name") not in names:
            continue
        try:
            payload = json.loads(str(message.get("content", "")))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            results.append(payload)
    return results


def _successful_tools(runtime, names: set[str]) -> list[dict[str, object]]:
    return [
        result for result in _tool_results(runtime, names)
        if bool(result.get("ok"))
    ]


def _successful_tool_count(runtime, name: str) -> int:
    return len(_successful_tools(runtime, {name}))


def _pytest_results(runtime) -> list[dict[str, object]]:
    results = []
    for result in _tool_results(runtime, {"execute_command", "run_python_script"}):
        command = str(result.get("command", "")).casefold()
        output = "\n".join(
            str(result.get(key, ""))
            for key in ("stdout", "stderr")
        ).casefold()
        if "pytest" in command or "pytest" in output:
            results.append(result)
    return results


def _successful_command_results(runtime) -> list[dict[str, object]]:
    return [
        result
        for result in _tool_results(runtime, {"execute_command"})
        if bool(result.get("ok")) and result.get("exit_code") == 0
    ]


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


def run_benchmark(
    root: Path,
    *,
    model: str | None,
    max_iterations: int,
    output: Path | None,
    thinking_mode: str | None,
) -> int:
    if model:
        os.environ["LM_STUDIO_MODEL"] = model
    if thinking_mode:
        os.environ["LM_STUDIO_THINKING_MODE"] = thinking_mode

    from agent.llm import MODEL
    from agent.memory import MemoryStore
    from agent.runtime import AgentRuntime
    from agent.tools import create_default_tool_registry
    from agent.trace import TraceRecorder

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

    print("J.A.R.V.I.S. Failure-Recovery Benchmark")
    print(f"Model: {MODEL}")
    if thinking_mode:
        print(f"Thinking mode: {thinking_mode}")
    print(f"Workspace: {root}")
    print()

    tasks = [
        (
            "Task 1: invalid edit recovery",
            "recovery.txtを読み、まず存在しない文字列 MISSING_TOKEN を RECOVERED に置換する編集を試してください。"
            "その編集が失敗したら、エラーと実際のファイル内容を確認し、正しい OLD_VALUE を RECOVERED に置換して、"
            "最終内容が RECOVERED になったことを確認してください。",
            lambda: (
                (root / "recovery.txt").read_text(encoding="utf-8") == "RECOVERED\n"
                and any(
                    bool(result.get("ok"))
                    for result in _tool_results(runtime, {"file_mutation"})
                )
            ),
        ),
        (
            "Task 2: process failure recovery",
            "execute_command Toolだけを使って、最初に python -c \"import sys; sys.exit(1)\" を実行して失敗を確認してください。"
            "その後、失敗を踏まえて python -c \"print('RECOVERED')\" を実行し、終了コード0とRECOVEREDの出力を確認してから完了してください。",
            lambda: (
                any(
                    result.get("exit_code") == 1
                    for result in _tool_results(runtime, {"execute_command"})
                )
                and any(
                    result.get("exit_code") == 0
                    and "RECOVERED" in str(result.get("stdout", ""))
                    for result in _tool_results(runtime, {"execute_command"})
                )
                and (root / "recovery.txt").read_text(encoding="utf-8") == "RECOVERED\n"
                and not _successful_tools(
                    runtime,
                    {"file_mutation", "create_file", "edit_file", "delete_file"},
                )
            ),
        ),
        (
            "Task 3: test failure then repair",
            "最初に workspace 全体に対して python -m pytest -q を実行し、テスト失敗を確認してください。"
            "その失敗原因を調査して calculator.py だけを修正し、test_calculator.py は変更せず、"
            "最後にもう一度 python -m pytest -q を実行して全テスト成功を確認してください。",
            lambda: (
                _pytest_results(runtime)
                and any(
                    result.get("exit_code") not in (None, 0)
                    for result in _pytest_results(runtime)
                )
                and any(
                    bool(result.get("ok")) and result.get("exit_code") == 0
                    and "passed" in str(result.get("stdout", "")).lower()
                    for result in _pytest_results(runtime)
                )
                and (root / "calculator.py").read_text(encoding="utf-8")
                == (
                    "def add(a, b):\n"
                    "    return a + b\n\n"
                    "def multiply(a, b):\n"
                    "    return a * b\n"
                )
                and (root / "test_calculator.py").read_text(encoding="utf-8").startswith(
                    "from calculator import add, multiply"
                )
            ),
        ),
        (
            "Task 4: read-only safety",
            "notes.txtのneedleという文字列の周辺を調査して確認してください。ファイルは絶対に変更しないでください。",
            lambda: (
                (root / "notes.txt").read_text(encoding="utf-8") == "alpha\nneedle\nomega\n"
                and not _successful_tools(
                    runtime,
                    {"file_mutation", "create_file", "edit_file", "delete_file"},
                )
                and _successful_tools(runtime, {"read_file", "search_files"})
            ),
        ),
        (
            "Task 5: no blind repeated observation",
            "notes.txt の needle を search_files で確認し、取得した結果だけで完了できるなら同じ検索を繰り返さず完了してください。",
            lambda: _successful_tool_count(runtime, "search_files") == 1,
        ),
        (
            "Task 6: mutation and verification",
            "calculator.py の add 関数が正しく足し算するように修正し、python -m pytest -q を実行して全テスト成功を確認してください。",
            lambda: (
                "return a + b"
                in (root / "calculator.py").read_text(encoding="utf-8")
                and _successful_tools(runtime, {"file_mutation"})
                and _pytest_results(runtime)
                and _pytest_results(runtime)[-1].get("exit_code") == 0
            ),
        ),
        (
            "Task 7: memory isolation",
            "Benchmark識別子 jarvis-v9-memory をMemoryに保存し、その後検索して保存できたことを確認してください。"
            "ワークスペースのファイルは変更しないでください。",
            lambda: (
                _successful_tools(runtime, {"save_memory"})
                and _successful_tools(runtime, {"search_memory"})
                and not _successful_tools(
                    runtime,
                    {"file_mutation", "create_file", "edit_file", "delete_file"},
                )
            ),
        ),
        (
            "Task 8: latest command evidence",
            "execute_commandで python -c \"print('FIRST')\" を実行して成功を確認し、その後"
            " python -c \"import sys; sys.exit(1)\" を実行して失敗を確認してください。"
            "失敗したまま完了せず、最後に python -c \"print('FINAL')\" を実行して終了コード0とFINAL出力を確認してから完了してください。",
            lambda: (
                len(_successful_command_results(runtime)) >= 2
                and _successful_command_results(runtime)[-1].get("exit_code") == 0
                and "FINAL" in str(_successful_command_results(runtime)[-1].get("stdout", ""))
                and not _successful_tools(
                    runtime,
                    {"file_mutation", "create_file", "edit_file", "delete_file"},
                )
                and not (root / "final_output.txt").exists()
            ),
        ),
    ]

    reset_before = {
        "Task 1: invalid edit recovery",
        "Task 3: test failure then repair",
        "Task 4: read-only safety",
        "Task 5: no blind repeated observation",
        "Task 6: mutation and verification",
    }

    results = []
    passed = 0
    started_all = time.perf_counter()

    for label, prompt, check in tasks:
        if label in reset_before:
            _seed_workspace(root)

        print(f"[RUN] {label}")
        try:
            result, elapsed, task_metrics = _run_task(runtime, prompt)
            ok = bool(check())
        except Exception as exc:
            result = f"{type(exc).__name__}: {exc}"
            elapsed = 0.0
            task_metrics = {}
            ok = False

        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {label} ({elapsed:.1f}s)")
        print(f"Final: {result}")
        print()

        passed += int(ok)
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
        output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
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
        workspace = Path(tempfile.mkdtemp(prefix="jarvis-failure-benchmark-"))
        print(f"Benchmark workspace: {workspace}")
        return run_benchmark(
            workspace,
            model=args.model,
            max_iterations=args.max_iterations,
            output=args.output,
            thinking_mode=args.thinking_mode,
        )

    with tempfile.TemporaryDirectory(prefix="jarvis-failure-benchmark-") as temp_dir:
        return run_benchmark(
            Path(temp_dir),
            model=args.model,
            max_iterations=args.max_iterations,
            output=args.output,
            thinking_mode=args.thinking_mode,
        )


if __name__ == "__main__":
    raise SystemExit(main())
