from __future__ import annotations

import argparse
import json
from pathlib import Path


METRICS = (
    "llm_calls",
    "tool_calls",
    "llm_duration_ms",
    "tool_duration_ms",
    "prompt_tokens",
    "completion_tokens",
    "reasoning_tokens",
    "prompt_chars",
    "tool_schema_chars",
)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def task_map(report: dict) -> dict[str, dict]:
    return {task["label"]: task for task in report.get("tasks", [])}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare JARVIS benchmark JSON reports."
    )
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--markdown", action="store_true")
    args = parser.parse_args()

    reports = [(path, load(path)) for path in args.reports]
    labels = sorted(
        {
            task["label"]
            for _, report in reports
            for task in report.get("tasks", [])
        }
    )

    print("J.A.R.V.I.S. Benchmark Comparison")
    print()
    for path, report in reports:
        metrics = report.get("metrics", {})
        passed = report.get("passed", 0)
        total = report.get("total_tasks", 0)
        print(
            f"{path.name}: {report.get('model', '?')} | "
            f"{passed}/{total} passed | "
            f"{report.get('total_elapsed_seconds', 0)}s | "
            f"LLM {metrics.get('llm_duration_ms', 0)}ms | "
            f"reasoning {metrics.get('reasoning_tokens', 0)}"
        )

    print()
    print("Per-task:")
    for label in labels:
        print(f"\n{label}")
        for path, report in reports:
            task = task_map(report).get(label)
            if not task:
                print(f"  {path.name}: missing")
                continue
            m = task.get("metrics", {})
            print(
                f"  {path.name}: "
                f"{'PASS' if task.get('passed') else 'FAIL'} | "
                f"{task.get('elapsed_seconds', 0)}s | "
                f"LLM {m.get('llm_calls', 0)} calls / "
                f"{m.get('llm_duration_ms', 0)}ms | "
                f"reasoning {m.get('reasoning_tokens', 0)} | "
                f"tools {m.get('tool_calls', 0)}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
