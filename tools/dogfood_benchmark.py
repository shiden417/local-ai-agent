from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run safe self-development Dogfooding tasks against a temporary copy of J.A.R.V.I.S."
    )
    parser.add_argument(
        "--model",
        help="LM Studio model ID. Defaults to LM_STUDIO_MODEL or the project default.",
    )
    parser.add_argument(
        "--thinking-mode",
        choices=("default", "think", "no_think"),
        help="Optional thinking mode. Qwen3 uses this setting.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=12,
        help="Maximum Agent iterations per task.",
    )
    parser.add_argument(
        "--keep-workspace",
        action="store_true",
        help="Keep the temporary Dogfooding workspace for inspection.",
    )
    return parser.parse_args()


def _copy_project(destination: Path) -> None:
    ignore = shutil.ignore_patterns(
        ".git",
        ".venv",
        "__pycache__",
        "*.pyc",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "benchmark-*.json",
        "trace.jsonl",
    )
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copytree(PROJECT_ROOT, destination, ignore=ignore, dirs_exist_ok=True)


def _inject_truncation_bug(workspace: Path) -> tuple[str, str]:
    path = workspace / "agent" / "observation.py"
    original = path.read_text(encoding="utf-8")
    buggy = original.replace(
        "return text[:head] + marker + text[-tail:], True",
        "return text[:head] + marker + text[-head:], True",
        1,
    )
    if buggy == original:
        raise RuntimeError("Failed to inject Dogfooding bug into observation.py")
    path.write_text(buggy, encoding="utf-8")
    return original, buggy


def _add_required_regression_test(workspace: Path) -> str:
    path = workspace / "tests" / "test_observation.py"
    original = path.read_text(encoding="utf-8")
    addition = """

def test_truncate_text_respects_max_chars_for_small_limits() -> None:
    text = "abcdefghijklmnopqrstuvwxyz" * 3

    result, truncated = truncate_text(text, max_chars=45)

    assert truncated is True
    assert len(result) == 45
    assert result.endswith("z")
"""
    path.write_text(original + addition, encoding="utf-8")
    return original + addition


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


def _successful_command(runtime) -> bool:
    return any(
        bool(result.get("ok")) and result.get("exit_code") == 0
        for result in _tool_results(runtime, {"execute_command", "run_python_script"})
    )


def _run_agent_task(runtime, prompt: str) -> str:
    result = runtime.run(prompt)
    return str(result)


def _task1_passed(
    workspace: Path,
    original_observation: str,
    original_tests: str,
    runtime,
) -> tuple[bool, str]:
    observation = (workspace / "agent" / "observation.py").read_text(encoding="utf-8")
    tests = (workspace / "tests" / "test_observation.py").read_text(encoding="utf-8")

    source_restored = observation == original_observation
    tests_preserved = tests == original_tests
    pytest_ok = _successful_command(runtime)
    passed_marker = any(
        "passed" in str(result.get("stdout", ""))
        for result in _tool_results(runtime, {"execute_command", "run_python_script"})
        if bool(result.get("ok"))
    )
    ok = source_restored and tests_preserved and pytest_ok and passed_marker
    details = (
        f"source_restored={source_restored}, "
        f"tests_preserved={tests_preserved}, "
        f"pytest_ok={pytest_ok}"
    )
    return ok, details


def _task2_passed(workspace: Path, original_completion_tests: str, runtime) -> tuple[bool, str]:
    path = workspace / "tests" / "test_completion_verifier.py"
    content = path.read_text(encoding="utf-8")
    test_added = (
        "def test_completion_verifier_allows_explicit_blocked_result" in content
        and content != original_completion_tests
    )
    pytest_ok = _successful_command(runtime)
    ok = test_added and pytest_ok
    return ok, f"test_added={test_added}, pytest_ok={pytest_ok}"


def run_dogfooding(
    workspace: Path,
    *,
    model: str | None,
    thinking_mode: str | None,
    max_iterations: int,
) -> int:
    if model:
        os.environ["LM_STUDIO_MODEL"] = model
    if thinking_mode:
        os.environ["LM_STUDIO_THINKING_MODE"] = thinking_mode

    from agent.runtime import AgentRuntime
    from agent.trace import TraceRecorder
    from agent.tools import create_default_tool_registry

    workspace.mkdir(parents=True, exist_ok=True)
    _copy_project(workspace)
    original_observation, _ = _inject_truncation_bug(workspace)
    original_tests = _add_required_regression_test(workspace)
    completion_tests = (workspace / "tests" / "test_completion_verifier.py").read_text(
        encoding="utf-8"
    )

    trace_path = workspace / "dogfood-trace.jsonl"
    runtime = AgentRuntime(
        working_directory=workspace,
        max_iterations=max_iterations,
        tool_registry=create_default_tool_registry(),
        confirm=lambda _message: True,
        trace_recorder=TraceRecorder(trace_path),
    )

    print("J.A.R.V.I.S. Safe Dogfooding")
    print(f"Workspace: {workspace}")
    print(f"Model: {os.getenv('LM_STUDIO_MODEL', 'project default')}")
    if thinking_mode:
        print(f"Thinking mode: {thinking_mode}")
    print()

    prompts = [
        (
            "Task 1: self-repair a failing regression",
            "このJ.A.R.V.I.S.プロジェクト自身を保守する開発Taskです。"
            "まずpytestを実行して失敗原因を調査してください。"
            "agent/observation.py に関係するtruncate_textの回帰テストが失敗しています。"
            "原因を確認して本番コード側を修正し、tests/test_observation.py は変更せず、"
            "python -m pytest -q を実行して全テスト成功を確認してください。"
            "テストを通すためだけにテストを書き換えないでください。",
        ),
        (
            "Task 2: add a focused regression test",
            "次の保守Taskです。"
            "agent/completion_verifier.py の現在の実装を確認し、"
            "finish_task が completion_status=blocked の明示的な結果を受け入れることを"
            "保証する回帰テストを tests/test_completion_verifier.py に1件追加してください。"
            "本番コードは変更せず、既存テストの意図を維持してください。"
            "追加後に python -m pytest -q を実行して全テスト成功を確認してください。",
        ),
    ]

    passed = 0
    for label, prompt in prompts:
        print(f"[RUN] {label}")
        try:
            _run_agent_task(runtime, prompt)
            if label.startswith("Task 1"):
                ok, details = _task1_passed(
                    workspace,
                    original_observation,
                    original_tests,
                    runtime,
                )
            else:
                ok, details = _task2_passed(
                    workspace,
                    completion_tests,
                    runtime,
                )
        except Exception as exc:
            ok = False
            details = f"{type(exc).__name__}: {exc}"

        print(f"[{'PASS' if ok else 'FAIL'}] {label}")
        print(details)
        print()
        if ok:
            passed += 1

    print(f"Result: {passed}/{len(prompts)} Dogfooding tasks passed")
    return 0 if passed == len(prompts) else 1


def main() -> int:
    args = _parse_args()
    if args.max_iterations < 1:
        raise SystemExit("--max-iterations must be at least 1")

    if args.keep_workspace:
        workspace = Path(tempfile.mkdtemp(prefix="jarvis-dogfood-"))
        print(f"Dogfooding workspace: {workspace}")
        return run_dogfooding(
            workspace,
            model=args.model,
            thinking_mode=args.thinking_mode,
            max_iterations=args.max_iterations,
        )

    with tempfile.TemporaryDirectory(prefix="jarvis-dogfood-") as temp_dir:
        return run_dogfooding(
            Path(temp_dir),
            model=args.model,
            thinking_mode=args.thinking_mode,
            max_iterations=args.max_iterations,
        )


if __name__ == "__main__":
    raise SystemExit(main())
