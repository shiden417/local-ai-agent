from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = PROJECT_ROOT / "tools" / "benchmark_agent.py"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the same J.A.R.V.I.S. benchmark for multiple LM Studio models."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["qwen/qwen3-8b", "google/gemma-4-e4b"],
        help="LM Studio model IDs to compare.",
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
        default="default",
        help="Use the same thinking-mode setting for every model."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark-compare.json"),
        help="Combined JSON report path.",
    )
    return parser.parse_args()


def _run_model(model: str, max_iterations: int, thinking_mode: str, work_dir: Path) -> dict[str, object]:
    report = work_dir / (model.replace("/", "_").replace(":", "_") + ".json")
    command = [
        sys.executable,
        str(BENCHMARK),
        "--model",
        model,
        "--thinking-mode",
        thinking_mode,
        "--max-iterations",
        str(max_iterations),
        "--output",
        str(report),
    ]
    completed = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )

    parsed: dict[str, object]
    if report.exists():
        try:
            parsed = json.loads(report.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            parsed = {"parse_error": str(exc), "report_status": "invalid_json"}
    else:
        parsed = {"report_status": "missing"}

    parsed["process_exit_code"] = completed.returncode
    parsed["stdout"] = completed.stdout[-8_000:]
    parsed["stderr"] = completed.stderr[-4_000:]
    parsed["requested_model"] = model
    parsed["thinking_mode"] = thinking_mode
    return parsed


def _print_result(result: dict[str, object]) -> None:
    model = result.get("requested_model", "unknown")
    process_exit_code = result.get("process_exit_code")
    passed = result.get("passed", "?")
    total = result.get("total_tasks", "?")
    metrics = result.get("metrics", {})
    llm_ms = metrics.get("llm_duration_ms", "?") if isinstance(metrics, dict) else "?"
    llm_calls = metrics.get("llm_calls", "?") if isinstance(metrics, dict) else "?"

    if process_exit_code == 0 and "passed" in result:
        print(f"{model}: {passed}/{total} tasks, LLM={llm_calls} calls, {llm_ms}ms")
        return

    print(f"{model}: benchmark failed (exit_code={process_exit_code})")
    if "parse_error" in result:
        print(f"  report parse error: {result['parse_error']}")
    elif result.get("report_status") == "missing":
        print("  benchmark JSON report was not created")

    stderr = str(result.get("stderr", "")).strip()
    stdout = str(result.get("stdout", "")).strip()
    if stderr:
        print("  stderr:")
        print(stderr[-2_000:])
    elif stdout:
        print("  stdout:")
        print(stdout[-4_000:])


def main() -> int:
    args = _parse_args()
    if args.max_iterations < 1:
        raise SystemExit("--max-iterations must be at least 1")
    if not args.models:
        raise SystemExit("--models must contain at least one model")

    with tempfile.TemporaryDirectory(prefix="jarvis-benchmark-compare-") as temp_dir:
        workspace = Path(temp_dir)
        results = [_run_model(model, args.max_iterations, args.thinking_mode, workspace) for model in args.models]

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "thinking_mode": args.thinking_mode,
        "max_iterations": args.max_iterations,
        "models": results,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("J.A.R.V.I.S. Model Comparison")
    for result in results:
        _print_result(result)

    print(f"JSON report: {output}")
    return 0 if all(result.get("process_exit_code") == 0 for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
