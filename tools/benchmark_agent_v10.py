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
        description="Run realistic multi-file JARVIS development tasks against LM Studio."
    )
    parser.add_argument("--model")
    parser.add_argument("--max-iterations", type=int, default=12)
    parser.add_argument(
        "--thinking-mode",
        choices=("default", "think", "no_think"),
    )
    parser.add_argument("--keep-workspace", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _seed_workspace(root: Path, *, broken_add: bool = True) -> None:
    src = root / "src"
    tests = root / "tests"
    src.mkdir(parents=True, exist_ok=True)
    tests.mkdir(parents=True, exist_ok=True)

    add_return = "a + 1" if broken_add else "a + b"
    (src / "calculator.py").write_text(
        "def add(a, b):\n"
        f"    return {add_return}\n\n"
        "def multiply(a, b):\n"
        "    return a * b\n",
        encoding="utf-8",
    )
    (tests / "test_calculator.py").write_text(
        "from src.calculator import add, multiply\n\n"
        "def test_add():\n"
        "    assert add(2, 3) == 5\n\n"
        "def test_multiply():\n"
        "    assert multiply(2, 3) == 6\n",
        encoding="utf-8",
    )
    (root / "app.py").write_text(
        "from src.calculator import add\n\n"
        "def main():\n"
        "    return add(2, 3)\n",
        encoding="utf-8",
    )
    (root / "config.json").write_text(
        '{"mode": "stable", "version": 1}\n',
        encoding="utf-8",
    )
    (root / "README.md").write_text(
        "# Calculator\n\nBasic calculator project.\n",
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


def _successful(runtime, names: set[str]) -> list[dict[str, object]]:
    return [x for x in _tool_results(runtime, names) if bool(x.get("ok"))]


def _commands(runtime) -> list[dict[str, object]]:
    return _tool_results(runtime, {"execute_command"})


def _pytest_results(runtime) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for result in _tool_results(runtime, {"execute_command", "run_python_script"}):
        command = str(result.get("command", "")).casefold()
        output = "\n".join(
            str(result.get(key, ""))
            for key in ("stdout", "stderr")
        ).casefold()
        if "pytest" in command or "pytest" in output:
            results.append(result)
    return results


def _has_successful_pytest(runtime) -> bool:
    results = _pytest_results(runtime)
    return bool(
        results
        and results[-1].get("exit_code") == 0
        and "passed" in str(results[-1].get("stdout", "")).casefold()
    )


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
    thinking_mode: str | None,
    output: Path | None,
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

    print("J.A.R.V.I.S. Realistic Development Benchmark")
    print(f"Model: {MODEL}")
    if thinking_mode:
        print(f"Thinking mode: {thinking_mode}")
    print(f"Workspace: {root}")
    print()

    expected_test_source = (
        "from src.calculator import add, multiply\n\n"
        "def test_add():\n"
        "    assert add(2, 3) == 5\n\n"
        "def test_multiply():\n"
        "    assert multiply(2, 3) == 6\n"
    )

    tasks = [
        (
            "Task 1: scoped bug fix",
            "src/calculator.py の add 関数のバグを調査して修正してください。"
            "tests/test_calculator.py は変更禁止です。修正後に python -m pytest -q を実行して全テスト成功を確認してください。",
            lambda: (
                (root / "src/calculator.py").read_text(encoding="utf-8")
                == "def add(a, b):\n    return a + b\n\ndef multiply(a, b):\n    return a * b\n"
                and (root / "tests/test_calculator.py").read_text(encoding="utf-8") == expected_test_source
                and (root / "app.py").read_text(encoding="utf-8").endswith("    return add(2, 3)\n")
                and (root / "config.json").read_text(encoding="utf-8") == '{"mode": "stable", "version": 1}\n'
                and (root / "README.md").read_text(encoding="utf-8") == "# Calculator\n\nBasic calculator project.\n"
                and _has_successful_pytest(runtime)
            ),
        ),
        (
            "Task 2: multi-file feature",
            "calculator に subtract(a, b) を追加してください。src/calculator.py と tests/test_calculator.py の両方を必要に応じて変更し、"
            "python -m pytest -q を実行して全テスト成功を確認してください。他のファイルは変更しないでください。",
            lambda: (
                "def subtract(a, b):" in (root / "src/calculator.py").read_text(encoding="utf-8")
                and "def test_subtract():" in (root / "tests/test_calculator.py").read_text(encoding="utf-8")
                and (root / "app.py").read_text(encoding="utf-8").endswith("    return add(2, 3)\n")
                and (root / "config.json").read_text(encoding="utf-8") == '{"mode": "stable", "version": 1}\n'
                and (root / "README.md").read_text(encoding="utf-8") == "# Calculator\n\nBasic calculator project.\n"
                and _has_successful_pytest(runtime)
            ),
        ),
        (
            "Task 3: documentation-only change",
            "README.md に subtract 関数の使い方を追記してください。コード、テスト、config.json、app.py は変更しないでください。"
            "README.mdだけを変更してください。",
            lambda: (
                "subtract" in (root / "README.md").read_text(encoding="utf-8").casefold()
                and (root / "src/calculator.py").read_text(encoding="utf-8").startswith("def add(a, b):")
                and (root / "tests/test_calculator.py").read_text(encoding="utf-8") == expected_test_source
                and (root / "config.json").read_text(encoding="utf-8") == '{"mode": "stable", "version": 1}\n'
                and (root / "app.py").read_text(encoding="utf-8").endswith("    return add(2, 3)\n")
            ),
        ),
        (
            "Task 4: process-only",
            "このworkspaceでファイルを変更せず、execute_command Toolを使って python -c \"print('JARVIS V10')\" を実行し、"
            "終了コード0と出力 JARVIS V10 を確認してください。",
            lambda: (
                any(
                    result.get("exit_code") == 0
                    and "JARVIS V10" in str(result.get("stdout", ""))
                    for result in _commands(runtime)
                )
                and not _successful(runtime, {"file_mutation", "create_file", "edit_file", "delete_file"})
            ),
        ),
        (
            "Task 5: failure-driven repair",
            "python -m pytest -q を実行して失敗を確認し、失敗原因を調査してください。"
            "src/calculator.pyだけを修正し、tests/test_calculator.pyは変更せず、最後にpython -m pytest -qを再実行して成功を確認してください。",
            lambda: (
                any(result.get("exit_code") not in (None, 0) for result in _pytest_results(runtime))
                and _has_successful_pytest(runtime)
                and "return a + b" in (root / "src/calculator.py").read_text(encoding="utf-8")
                and (root / "tests/test_calculator.py").read_text(encoding="utf-8") == expected_test_source
            ),
        ),
        (
            "Task 6: targeted investigation",
            "workspaceを調査し、config.json の mode と version の値だけ確認してください。"
            "ファイルは変更せず、無関係なファイルの内容を読み込まないでください。",
            lambda: (
                not _successful(runtime, {"file_mutation", "create_file", "edit_file", "delete_file"})
                and _successful(runtime, {"read_file", "search_files"})
                and all(
                    (
                        str(result.get("path", "")).replace("\\", "/").casefold().endswith("config.json")
                        or all(
                            str(match.get("path", "")).replace("\\", "/").casefold().endswith("config.json")
                            for match in result.get("matches", [])
                        )
                    )
                    for result in _successful(runtime, {"read_file", "search_files"})
                )
            ),
        ),
        (
            "Task 7: memory isolation",
            "Benchmark識別子 jarvis-v10-memory をMemoryに保存し、その後検索して保存できたことを確認してください。"
            "workspaceのファイルは変更しないでください。",
            lambda: (
                _successful(runtime, {"save_memory"})
                and _successful(runtime, {"search_memory"})
                and not _successful(runtime, {"file_mutation", "create_file", "edit_file", "delete_file"})
            ),
        ),
        (
            "Task 8: controlled rename",
            "multiply 関数を product にリネームしてください。src/calculator.py と tests/test_calculator.py の参照だけを必要に応じて変更し、"
            "app.py、config.json、README.md は変更しないでください。python -m pytest -q を実行して全テスト成功を確認してください。",
            lambda: (
                "def product(a, b):" in (root / "src/calculator.py").read_text(encoding="utf-8")
                and "multiply" not in (root / "src/calculator.py").read_text(encoding="utf-8")
                and "from src.calculator import add, product" in (root / "tests/test_calculator.py").read_text(encoding="utf-8")
                and "def test_product():" in (root / "tests/test_calculator.py").read_text(encoding="utf-8")
                and "multiply" not in (root / "tests/test_calculator.py").read_text(encoding="utf-8")
                and (root / "app.py").read_text(encoding="utf-8").endswith("    return add(2, 3)\n")
                and (root / "config.json").read_text(encoding="utf-8") == '{"mode": "stable", "version": 1}\n'
                and (root / "README.md").read_text(encoding="utf-8") == "# Calculator\n\nBasic calculator project.\n"
                and _has_successful_pytest(runtime)
            ),
        ),
    ]

    reset_before = {
        "Task 1: scoped bug fix",
        "Task 2: multi-file feature",
        "Task 3: documentation-only change",
        "Task 4: process-only",
        "Task 5: failure-driven repair",
        "Task 6: targeted investigation",
        "Task 7: memory isolation",
        "Task 8: controlled rename",
    }
    correct_baseline = {
        "Task 2: multi-file feature",
        "Task 8: controlled rename",
    }

    results: list[dict[str, object]] = []
    passed = 0
    started_all = time.perf_counter()

    for label, prompt, check in tasks:
        if label in reset_before:
            _seed_workspace(root, broken_add=label not in correct_baseline)
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
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"JSON report: {output}")

    print(
        f"Metrics: LLM={metrics['llm_calls']} calls, Tool={metrics['tool_calls']} calls, "
        f"LLM={metrics['llm_duration_ms']}ms, Reasoning={metrics['reasoning_tokens']} tokens"
    )
    print(f"Result: {passed}/{len(tasks)} tasks passed")
    return 0 if passed == len(tasks) else 1


def main() -> int:
    args = _parse_args()
    if args.max_iterations < 1:
        raise SystemExit("--max-iterations must be at least 1")

    if args.keep_workspace:
        workspace = Path(tempfile.mkdtemp(prefix="jarvis-dev-benchmark-"))
        print(f"Benchmark workspace: {workspace}")
        return run_benchmark(
            workspace,
            model=args.model,
            max_iterations=args.max_iterations,
            thinking_mode=args.thinking_mode,
            output=args.output,
        )

    with tempfile.TemporaryDirectory(prefix="jarvis-dev-benchmark-") as temp_dir:
        return run_benchmark(
            Path(temp_dir),
            model=args.model,
            max_iterations=args.max_iterations,
            thinking_mode=args.thinking_mode,
            output=args.output,
        )


if __name__ == "__main__":
    raise SystemExit(main())
