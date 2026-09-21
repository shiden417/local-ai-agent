from __future__ import annotations

from tools.compare_benchmarks import task_map


def test_task_map_indexes_benchmark_tasks() -> None:
    report = {
        "tasks": [
            {"label": "Task 1", "passed": True},
            {"label": "Task 2", "passed": False},
        ]
    }

    mapped = task_map(report)

    assert mapped["Task 1"]["passed"] is True
    assert mapped["Task 2"]["passed"] is False
