from __future__ import annotations

from pathlib import Path

from tools.benchmark_agent import _check_exact, _seed_workspace


def test_benchmark_workspace_seed_is_deterministic(tmp_path: Path) -> None:
    _seed_workspace(tmp_path)

    assert _check_exact(
        tmp_path / "calculator.py",
        "def add(a, b):\n"
        "    return a - b\n\n"
        "def multiply(a, b):\n"
        "    return a * b\n",
    )
    assert _check_exact(
        tmp_path / "test_calculator.py",
        "from calculator import add, multiply\n\n"
        "def test_add():\n"
        "    assert add(2, 3) == 5\n\n"
        "def test_multiply():\n"
        "    assert multiply(2, 3) == 6\n",
    )
