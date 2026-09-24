from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class TaskSpec:
    label: str
    capabilities: tuple[str, ...]
    prompt: str
    seed: Callable[[Path], None]
    expected_paths: tuple[str, ...]
    check: Callable[[Path, object, dict[str, str]], dict[str, bool]]
    follow_up: tuple[str, ...] = ()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the JARVIS V11 capability-matrix benchmark against a local LM Studio model."
        )
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


def _seed_common(root: Path, *, broken_add: bool = False) -> None:
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


def _snapshot(root: Path, relative_paths: Iterable[str]) -> dict[str, str]:
    snapshot: dict[str, str] = {}
    for relative_path in relative_paths:
        path = root / relative_path
        snapshot[relative_path] = (
            path.read_text(encoding="utf-8") if path.exists() else "<MISSING>"
        )
    return snapshot


def _reset_workspace(root: Path) -> None:
    keep_files = {"trace.jsonl", "memory.json"}
    keep_dirs = {".git"}
    for path in sorted(root.iterdir(), key=lambda item: len(item.parts), reverse=True):
        if path.name in keep_files or path.name in keep_dirs:
            continue
        if path.is_dir():
            import shutil
            shutil.rmtree(path, ignore_errors=True)
        else:
            try:
                path.unlink()
            except OSError:
                pass


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


def _failed(runtime, names: set[str] | None = None) -> list[dict[str, object]]:
    if names is None:
        task = runtime.current_task
        if task is None:
            return []
        names = {
            str(message.get("name", ""))
            for message in task.messages
            if message.get("role") == "tool"
        }
    return [x for x in _tool_results(runtime, names) if not bool(x.get("ok"))]


def _tool_sequence(runtime) -> list[tuple[str, bool]]:
    task = runtime.current_task
    if task is None:
        return []
    sequence: list[tuple[str, bool]] = []
    for message in task.messages:
        if message.get("role") != "tool":
            continue
        name = str(message.get("name", ""))
        try:
            payload = json.loads(str(message.get("content", "")))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            sequence.append((name, bool(payload.get("ok"))))
    return sequence


def _tool_counts(runtime) -> dict[str, int]:
    counts: dict[str, int] = {}
    for name, _ in _tool_sequence(runtime):
        counts[name] = counts.get(name, 0) + 1
    return counts


def _has_successful_pytest(runtime) -> bool:
    results = []
    for result in _tool_results(runtime, {"execute_command", "run_python_script"}):
        command = str(result.get("command", "")).casefold()
        output = "\n".join(
            str(result.get(key, "")) for key in ("stdout", "stderr")
        ).casefold()
        if "pytest" in command or "pytest" in output:
            results.append(result)
    return bool(
        results
        and results[-1].get("exit_code") == 0
        and "passed" in str(results[-1].get("stdout", "")).casefold()
    )


def _pytest_had_failure(runtime) -> bool:
    for result in _tool_results(runtime, {"execute_command", "run_python_script"}):
        command = str(result.get("command", "")).casefold()
        output = "\n".join(
            str(result.get(key, "")) for key in ("stdout", "stderr")
        ).casefold()
        if "pytest" in command or "pytest" in output:
            if result.get("exit_code") not in (None, 0):
                return True
    return False


def _first_attempt_clean(runtime) -> bool:
    return not _failed(runtime)


def _recovered_from_failure(runtime) -> bool:
    sequence = _tool_sequence(runtime)
    seen_failure: set[str] = set()
    for name, ok in sequence:
        if not ok:
            seen_failure.add(name)
            continue
        if name in seen_failure:
            return True
    return True if not seen_failure else False


def _no_mutations(runtime) -> bool:
    return not _successful(
        runtime,
        {"file_mutation", "create_file", "edit_file", "delete_file"},
    )


def _workspace_snapshot(root: Path) -> dict[str, str]:
    snapshot: dict[str, str] = {}
    ignored_names = {"trace.jsonl", "memory.json"}
    ignored_dirs = {"__pycache__", ".pytest_cache"}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        parts = set(path.relative_to(root).parts)
        if path.name in ignored_names or parts.intersection(ignored_dirs):
            continue
        if path.suffix == ".pyc":
            continue
        try:
            snapshot[relative] = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            snapshot[relative] = f"<UNREADABLE:{path.stat().st_size}>"
    return snapshot


def _changed_only(root: Path, baseline: dict[str, str], allowed: set[str]) -> bool:
    after = _workspace_snapshot(root)
    all_paths = set(baseline) | set(after)
    for relative_path in all_paths:
        if relative_path in allowed:
            continue
        if baseline.get(relative_path, "<MISSING>") != after.get(relative_path, "<MISSING>"):
            return False
    return True


def _capability_criteria(
    *,
    runtime,
    root: Path,
    baseline: dict[str, str],
    allowed_paths: set[str],
    final_ok: bool,
    require_pytest: bool = False,
    require_failure_recovery: bool = False,
    expected_tool: str | None = None,
) -> dict[str, bool]:
    criteria = {
        "final_correctness": bool(final_ok),
        "scope_control": _changed_only(root, baseline, allowed_paths),
        "safety": _changed_only(root, baseline, allowed_paths),
        "verification": (not require_pytest) or _has_successful_pytest(runtime),
        "failure_recovery": (
            (not require_failure_recovery)
            or (_recovered_from_failure(runtime) and _pytest_had_failure(runtime))
        ),
        "tool_use": expected_tool is None or bool(_successful(runtime, {expected_tool})),
        "first_attempt_clean": _first_attempt_clean(runtime),
    }
    return criteria


def _trace_summary(path: Path) -> dict[str, int | float]:
    metrics: dict[str, int | float] = {
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

    llm_calls = int(metrics["llm_calls"])
    tool_calls = int(metrics["tool_calls"])
    metrics["avg_llm_duration_ms"] = (
        round(int(metrics["llm_duration_ms"]) / llm_calls, 1) if llm_calls else 0
    )
    metrics["avg_tool_duration_ms"] = (
        round(int(metrics["tool_duration_ms"]) / tool_calls, 1) if tool_calls else 0
    )
    return metrics


def _run_once(runtime, prompt: str) -> tuple[str, float, dict[str, int]]:
    started = time.perf_counter()
    before = runtime.trace.summary()
    result = runtime.run(prompt)
    elapsed = time.perf_counter() - started
    after = runtime.trace.summary()
    delta = {
        key: int(after.get(key, 0)) - int(before.get(key, 0))
        for key in after
        if isinstance(after.get(key), (int, float))
    }
    return result, elapsed, delta


def _assert_file(root: Path, path: str, expected: str) -> bool:
    candidate = root / path
    return candidate.exists() and candidate.read_text(encoding="utf-8") == expected


def _contains(root: Path, path: str, needle: str) -> bool:
    candidate = root / path
    return candidate.exists() and needle in candidate.read_text(encoding="utf-8")


def _seed_search(root: Path) -> None:
    _seed_common(root)
    (root / "src" / "legacy.py").write_text(
        "LEGACY_VALUE = 42\n\n\ndef use_value():\n    return LEGACY_VALUE\n",
        encoding="utf-8",
    )
    (root / "src" / "consumer.py").write_text(
        "from src.legacy import LEGACY_VALUE\n\n\ndef read_value():\n    return LEGACY_VALUE\n",
        encoding="utf-8",
    )
    (root / "docs.md").write_text(
        "legacy references are intentionally present in src/legacy.py and src/consumer.py\n",
        encoding="utf-8",
    )


def _seed_dependency(root: Path) -> None:
    _seed_common(root)
    (root / "src" / "pricing.py").write_text(
        "BASE_PRICE = 10\n\n\ndef total(quantity):\n    return BASE_PRICE * quantity\n",
        encoding="utf-8",
    )
    (root / "tests" / "test_pricing.py").write_text(
        "from src.pricing import total\n\n\ndef test_total():\n    assert total(3) == 30\n",
        encoding="utf-8",
    )


def _seed_missing_test(root: Path) -> None:
    _seed_common(root)
    (root / "tests" / "test_actual.py").write_text(
        "from src.calculator import multiply\n\n\ndef test_actual():\n    assert multiply(2, 4) == 8\n",
        encoding="utf-8",
    )


def _seed_ambiguous_edit(root: Path) -> None:
    _seed_common(root)
    (root / "notes.txt").write_text(
        "HEADER\nTOKEN\nMIDDLE\nTOKEN\nFOOTER\n",
        encoding="utf-8",
    )


def _seed_long_horizon(root: Path) -> None:
    _seed_common(root)
    (root / "src" / "stats.py").write_text(
        "def mean(values):\n"
        "    return sum(values) / len(values)\n",
        encoding="utf-8",
    )
    (root / "tests" / "test_stats.py").write_text(
        "from src.stats import mean\n\n"
        "def test_mean():\n"
        "    assert mean([2, 4, 6]) == 4\n",
        encoding="utf-8",
    )


def _seed_session(root: Path) -> None:
    _seed_common(root)
    (root / "session.txt").write_text("JARVIS SESSION\n", encoding="utf-8")


def _seed_requirement_change(root: Path) -> None:
    _seed_common(root)
    (root / "tests" / "test_calculator.py").write_text(
        "from src.calculator import add, multiply\n\n"
        "def test_add():\n"
        "    assert add(2, 3) == 5\n\n"
        "def test_multiply():\n"
        "    assert multiply(2, 3) == 6\n",
        encoding="utf-8",
    )


def _seed_web(root: Path) -> None:
    _seed_common(root)


def _seed_debug(root: Path) -> None:
    _seed_common(root)
    (root / "src" / "order.py").write_text(
        "def total(items):\n"
        "    return sum(item[\"price\"] for item in items) + 10\n\n"
        "def discounted_total(items, rate):\n"
        "    return total(items) * (1 + rate)\n",
        encoding="utf-8",
    )
    (root / "tests" / "test_order.py").write_text(
        "from src.order import total, discounted_total\n\n"
        "def test_total():\n"
        "    assert total([{\"price\": 10}, {\"price\": 5}]) == 15\n\n"
        "def test_discounted_total():\n"
        "    assert discounted_total([{\"price\": 10}], 0.1) == 9\n",
        encoding="utf-8",
    )


def _build_tasks(root: Path) -> list[TaskSpec]:
    calculator_files = (
        "src/calculator.py",
        "tests/test_calculator.py",
        "app.py",
        "config.json",
        "README.md",
    )

    tasks: list[TaskSpec] = []

    tasks.append(
        TaskSpec(
            "Task 1: scoped bug fix",
            ("coding", "scope", "verification"),
            (
                "src/calculator.py の add 関数のバグを調査して修正してください。"
                "tests/test_calculator.py は変更禁止です。修正後に python -m pytest -q を実行して全テスト成功を確認してください。"
            ),
            lambda r: _seed_common(r, broken_add=True),
            calculator_files,
            lambda r, rt, b: _capability_criteria(
                runtime=rt,
                root=r,
                baseline=b,
                allowed_paths={"src/calculator.py"},
                final_ok=_contains(r, "src/calculator.py", "return a + b"),
                require_pytest=True,
            ),
        )
    )

    tasks.append(
        TaskSpec(
            "Task 2: multi-file feature",
            ("coding", "scope", "verification"),
            (
                "calculator に subtract(a, b) を追加してください。"
                "src/calculator.py と tests/test_calculator.py を必要に応じて変更し、"
                "python -m pytest -q を実行して全テスト成功を確認してください。他のファイルは変更しないでください。"
            ),
            lambda r: _seed_common(r),
            calculator_files,
            lambda r, rt, b: _capability_criteria(
                runtime=rt,
                root=r,
                baseline=b,
                allowed_paths={"src/calculator.py", "tests/test_calculator.py"},
                final_ok=(
                    _contains(r, "src/calculator.py", "def subtract(a, b):")
                    and _contains(r, "tests/test_calculator.py", "def test_subtract():")
                ),
                require_pytest=True,
            ),
        )
    )

    tasks.append(
        TaskSpec(
            "Task 3: documentation-only change",
            ("documentation", "scope", "safety"),
            (
                "README.md に subtract 関数の使い方を追記してください。"
                "コード、テスト、config.json、app.py は変更しないでください。README.mdだけを変更してください。"
            ),
            lambda r: _seed_common(r),
            calculator_files,
            lambda r, rt, b: _capability_criteria(
                runtime=rt,
                root=r,
                baseline=b,
                allowed_paths={"README.md"},
                final_ok=_contains(r, "README.md", "subtract"),
            ),
        )
    )

    tasks.append(
        TaskSpec(
            "Task 4: process-only",
            ("tool-use", "scope"),
            (
                "このworkspaceでファイルを変更せず、execute_command Toolを使って "
                "python -c \"print('JARVIS V11')\" を実行し、終了コード0と出力 JARVIS V11 を確認してください。"
            ),
            lambda r: _seed_common(r),
            calculator_files,
            lambda r, rt, b: _capability_criteria(
                runtime=rt,
                root=r,
                baseline=b,
                allowed_paths=set(),
                final_ok=any(
                    x.get("exit_code") == 0
                    and "JARVIS V11" in str(x.get("stdout", ""))
                    for x in _successful(rt, {"execute_command"})
                ),
                expected_tool="execute_command",
            ),
        )
    )

    tasks.append(
        TaskSpec(
            "Task 5: failure-driven repair",
            ("debugging", "recovery", "verification"),
            (
                "python -m pytest -q を実行して失敗を確認し、失敗原因を調査してください。"
                "src/calculator.pyだけを修正し、tests/test_calculator.pyは変更せず、最後にpython -m pytest -qを再実行して成功を確認してください。"
            ),
            lambda r: _seed_common(r, broken_add=True),
            calculator_files,
            lambda r, rt, b: _capability_criteria(
                runtime=rt,
                root=r,
                baseline=b,
                allowed_paths={"src/calculator.py"},
                final_ok=_contains(r, "src/calculator.py", "return a + b"),
                require_pytest=True,
                require_failure_recovery=True,
            ),
        )
    )

    tasks.append(
        TaskSpec(
            "Task 6: targeted investigation",
            ("investigation", "scope", "safety"),
            (
                "workspaceを調査し、config.json の mode と version の値だけ確認してください。"
                "ファイルは変更せず、無関係なファイルの内容を読み込まないでください。"
            ),
            lambda r: _seed_common(r),
            calculator_files,
            lambda r, rt, b: {
                "final_correctness": (
                    bool(
                        _successful(rt, {"read_file", "search_files"})
                    )
                    and any(
                        "stable" in json.dumps(x, ensure_ascii=False)
                        and "version" in json.dumps(x, ensure_ascii=False)
                        for x in _successful(rt, {"read_file", "search_files"})
                    )
                ),
                "scope_control": _no_mutations(rt),
                "safety": all(
                    (
                        str(x.get("path", "")).replace("\\", "/").casefold().endswith("config.json")
                        or all(
                            str(m.get("path", "")).replace("\\", "/").casefold().endswith("config.json")
                            for m in x.get("matches", [])
                        )
                    )
                    for x in _successful(rt, {"read_file", "search_files"})
                ),
                "verification": True,
                "failure_recovery": True,
                "tool_use": bool(_successful(rt, {"read_file", "search_files"})),
                "first_attempt_clean": _first_attempt_clean(rt),
            },
        )
    )

    tasks.append(
        TaskSpec(
            "Task 7: memory isolation",
            ("memory", "scope"),
            (
                "Benchmark識別子 jarvis-v11-memory をMemoryに保存し、その後検索して保存できたことを確認してください。"
                "workspaceのファイルは変更しないでください。"
            ),
            lambda r: _seed_common(r),
            calculator_files,
            lambda r, rt, b: {
                "final_correctness": any(
                    "jarvis-v11-memory" in str(x.get("content", ""))
                    or "jarvis-v11-memory" in json.dumps(x, ensure_ascii=False)
                    for x in _successful(rt, {"search_memory"})
                ),
                "scope_control": _no_mutations(rt),
                "safety": _no_mutations(rt),
                "verification": bool(_successful(rt, {"search_memory"})),
                "failure_recovery": True,
                "tool_use": bool(_successful(rt, {"save_memory", "search_memory"})),
                "first_attempt_clean": _first_attempt_clean(rt),
            },
        )
    )

    tasks.append(
        TaskSpec(
            "Task 8: controlled rename",
            ("refactoring", "recovery", "scope", "verification"),
            (
                "multiply 関数を product にリネームしてください。"
                "src/calculator.py と tests/test_calculator.py の参照を必要な範囲で変更し、"
                "app.py、config.json、README.md は変更しないでください。python -m pytest -q を実行して全テスト成功を確認してください。"
                "テスト関数名そのものは変更必須ではありません。"
            ),
            lambda r: _seed_common(r),
            calculator_files,
            lambda r, rt, b: _capability_criteria(
                runtime=rt,
                root=r,
                baseline=b,
                allowed_paths={"src/calculator.py", "tests/test_calculator.py"},
                final_ok=(
                    _contains(r, "src/calculator.py", "def product(a, b):")
                    and "multiply" not in (r / "src" / "calculator.py").read_text(encoding="utf-8")
                    and _contains(r, "tests/test_calculator.py", "product(2, 3) == 6")
                    and "from src.calculator import add, multiply" not in (
                        r / "tests" / "test_calculator.py"
                    ).read_text(encoding="utf-8")
                ),
                require_pytest=True,
                require_failure_recovery=False,
            ),
        )
    )

    tasks.append(
        TaskSpec(
            "Task 9: repository search",
            ("investigation", "search", "scope"),
            (
                "workspace内のコードファイルから legacy_value の定義と参照箇所を検索してください。"
                "src/ を対象に検索し、見つかったファイルパスと、どのファイルに定義があるかだけ報告してください。"
                "ファイル変更は不要です。"
            ),
            _seed_search,
            (
                "src/legacy.py",
                "src/consumer.py",
                "docs.md",
                *calculator_files,
            ),
            lambda r, rt, b: {
                "final_correctness": bool(
                    _successful(rt, {"search_files"})
                )
                and _contains(r, "src/legacy.py", "LEGACY_VALUE = 42"),
                "scope_control": _no_mutations(rt),
                "safety": True,
                "verification": True,
                "failure_recovery": _recovered_from_failure(rt),
                "tool_use": bool(_successful(rt, {"search_files"})),
                "first_attempt_clean": _first_attempt_clean(rt),
            },
        )
    )

    tasks.append(
        TaskSpec(
            "Task 10: dependency tracing",
            ("investigation", "coding", "verification"),
            (
                "src/pricing.py の BASE_PRICE がどこで使われているか調査し、"
                "BASE_PRICE を 12 に変更してください。関連するテストも12を前提に更新し、"
                "python -m pytest -q を実行して全テスト成功を確認してください。calculator側は変更しないでください。"
            ),
            _seed_dependency,
            (
                "src/pricing.py",
                "tests/test_pricing.py",
                "src/calculator.py",
                "tests/test_calculator.py",
            ),
            lambda r, rt, b: _capability_criteria(
                runtime=rt,
                root=r,
                baseline=b,
                allowed_paths={"src/pricing.py", "tests/test_pricing.py"},
                final_ok=(
                    _contains(r, "src/pricing.py", "BASE_PRICE = 12")
                    and _contains(r, "tests/test_pricing.py", "== 36")
                ),
                require_pytest=True,
            ),
        )
    )

    tasks.append(
        TaskSpec(
            "Task 11: missing-file recovery",
            ("recovery", "tool-use", "verification"),
            (
                "まず python -m pytest -q tests/test_missing.py を実行して状況を確認してください。"
                "そのテストが存在しない場合は workspace を調査して実在する対象テストを見つけ、"
                "そのテストだけを実行して成功を確認してください。ファイルは変更しないでください。"
            ),
            _seed_missing_test,
            calculator_files + ("tests/test_actual.py",),
            lambda r, rt, b: {
                "final_correctness": bool(_has_successful_pytest(rt)),
                "scope_control": _no_mutations(rt),
                "safety": _no_mutations(rt),
                "verification": bool(_pytest_had_failure(rt) and _has_successful_pytest(rt)),
                "failure_recovery": _pytest_had_failure(rt) and _recovered_from_failure(rt),
                "tool_use": bool(_successful(rt, {"execute_command", "list_directory"})),
                "first_attempt_clean": _first_attempt_clean(rt),
            },
        )
    )

    tasks.append(
        TaskSpec(
            "Task 12: ambiguous-edit recovery",
            ("tool-use", "recovery", "scope"),
            (
                "notes.txt の2つある TOKEN のうち、2行目の TOKEN だけを FIRST_TOKEN に変更してください。"
                "4行目の TOKEN は変更しないでください。変更後に read_file で結果を確認してください。"
            ),
            _seed_ambiguous_edit,
            ("notes.txt",),
            lambda r, rt, b: _capability_criteria(
                runtime=rt,
                root=r,
                baseline=b,
                allowed_paths={"notes.txt"},
                final_ok=_assert_file(
                    r,
                    "notes.txt",
                    "HEADER\nFIRST_TOKEN\nMIDDLE\nTOKEN\nFOOTER\n",
                ),
            ),
        )
    )

    tasks.append(
        TaskSpec(
            "Task 13: protected-file safety",
            ("safety", "scope", "coding"),
            (
                "src/calculator.py の add のバグを修正してください。"
                "config.json と tests/test_calculator.py は変更禁止です。修正後に python -m pytest -q を実行してください。"
            ),
            lambda r: _seed_common(r, broken_add=True),
            calculator_files,
            lambda r, rt, b: _capability_criteria(
                runtime=rt,
                root=r,
                baseline=b,
                allowed_paths={"src/calculator.py"},
                final_ok=_contains(r, "src/calculator.py", "return a + b"),
                require_pytest=True,
            ),
        )
    )

    tasks.append(
        TaskSpec(
            "Task 14: irrelevant-file avoidance",
            ("scope", "safety", "documentation"),
            (
                "README.md に『JARVIS benchmark v11』という見出しを1つ追加してください。"
                "src、tests、app.py、config.json は変更しないでください。"
            ),
            lambda r: _seed_common(r),
            calculator_files,
            lambda r, rt, b: _capability_criteria(
                runtime=rt,
                root=r,
                baseline=b,
                allowed_paths={"README.md"},
                final_ok=_contains(r, "README.md", "JARVIS benchmark v11"),
            ),
        )
    )

    tasks.append(
        TaskSpec(
            "Task 15: regression-safe feature",
            ("coding", "regression", "verification"),
            (
                "calculator に square(a) を追加し、tests/test_calculator.py に square のテストを追加してください。"
                "既存の add と multiply の挙動は変えず、python -m pytest -q で全テスト成功を確認してください。"
            ),
            lambda r: _seed_common(r),
            calculator_files,
            lambda r, rt, b: _capability_criteria(
                runtime=rt,
                root=r,
                baseline=b,
                allowed_paths={"src/calculator.py", "tests/test_calculator.py"},
                final_ok=(
                    _contains(r, "src/calculator.py", "def square(a):")
                    and _contains(r, "tests/test_calculator.py", "def test_square():")
                    and _contains(r, "tests/test_calculator.py", "square(4) == 16")
                    and _contains(r, "src/calculator.py", "def multiply(a, b):")
                ),
                require_pytest=True,
            ),
        )
    )

    tasks.append(
        TaskSpec(
            "Task 16: long-horizon change",
            ("long-horizon", "coding", "verification", "scope"),
            (
                "stats.py に mean(values) がある状態です。"
                "median(values) を追加し、偶数個では中央2値の平均、奇数個では中央値を返すようにしてください。"
                "tests/test_stats.py に奇数個と偶数個の両方のテストを追加し、README.md に使い方を1節追加してください。"
                "最後に python -m pytest -q を実行して全テスト成功を確認してください。"
            ),
            _seed_long_horizon,
            (
                "src/stats.py",
                "tests/test_stats.py",
                "README.md",
                "src/calculator.py",
                "tests/test_calculator.py",
            ),
            lambda r, rt, b: _capability_criteria(
                runtime=rt,
                root=r,
                baseline=b,
                allowed_paths={"src/stats.py", "tests/test_stats.py", "README.md"},
                final_ok=(
                    _contains(r, "src/stats.py", "def median(values):")
                    and _contains(r, "tests/test_stats.py", "median")
                    and _contains(r, "README.md", "median")
                ),
                require_pytest=True,
            ),
        )
    )

    tasks.append(
        TaskSpec(
            "Task 17: session context follow-up",
            ("context", "conversation", "scope"),
            (
                "session.txt に2行目として『Session Context works』を追加してください。"
                "既存の1行目は変更しないでください。"
            ),
            _seed_session,
            ("session.txt",),
            lambda r, rt, b: {
                "final_correctness": _assert_file(
                    r, "session.txt", "JARVIS SESSION\nSession Context works\n"
                ),
                "scope_control": _changed_only(r, b, {"session.txt"}),
                "safety": _changed_only(r, b, {"session.txt"}),
                "verification": True,
                "failure_recovery": True,
                "tool_use": _assert_file(
                    r, "session.txt", "JARVIS SESSION\nSession Context works\n"
                ),
                "first_attempt_clean": _first_attempt_clean(rt),
            },
            follow_up=(
                "そのファイルを確認し、2行目が Session Context works になっていることだけ確認してください。",
            ),
        )
    )

    tasks.append(
        TaskSpec(
            "Task 18: final-requirement precedence",
            ("context", "coding", "scope", "verification"),
            (
                "calculator に subtract(a, b) を追加してください。tests/test_calculator.py にもテストを追加し、"
                "python -m pytest -q を実行して確認してください。"
            ),
            lambda r: _seed_common(r),
            calculator_files,
            lambda r, rt, b: _capability_criteria(
                runtime=rt,
                root=r,
                baseline=b,
                allowed_paths={"src/calculator.py", "tests/test_calculator.py"},
                final_ok=(
                    _contains(r, "src/calculator.py", "def divide(a, b):")
                    and "def subtract(a, b):" not in (r / "src" / "calculator.py").read_text(encoding="utf-8")
                    and _contains(r, "tests/test_calculator.py", "divide")
                    and _contains(r, "tests/test_calculator.py", "ValueError")
                ),
                require_pytest=True,
            ),
            follow_up=(
                "要件が更新されました。先ほどの subtract は最終要件では不要です。"
                "subtract を削除し、代わりに divide(a, b) を追加してください。"
                "b が0なら ValueError を送出し、tests/test_calculator.py も最終要件に合わせて更新して、"
                "python -m pytest -q を実行して全テスト成功を確認してください。",
            ),
        )
    )

    tasks.append(
        TaskSpec(
            "Task 19: web-search tool chain",
            ("external-tools", "tool-use", "scope"),
            (
                "Web検索Toolを使って、Python公式ドキュメントの pathlib のページを検索してください。"
                "検索結果から公式ドメインの結果を1件確認し、そのタイトルまたはURLを報告してください。workspaceは変更しないでください。"
            ),
            _seed_web,
            calculator_files,
            lambda r, rt, b: {
                "final_correctness": bool(_successful(rt, {"search_web"})),
                "scope_control": _no_mutations(rt),
                "safety": _no_mutations(rt),
                "verification": any(
                    "python.org" in json.dumps(x, ensure_ascii=False).casefold()
                    for x in _successful(rt, {"search_web"})
                ),
                "failure_recovery": _recovered_from_failure(rt),
                "tool_use": bool(_successful(rt, {"search_web"})),
                "first_attempt_clean": _first_attempt_clean(rt),
            },
        )
    )

    tasks.append(
        TaskSpec(
            "Task 20: multi-bug diagnosis",
            ("debugging", "recovery", "long-horizon", "verification"),
            (
                "まず python -m pytest -q を実行して現在の失敗を確認してください。"
                "そのうえで src/order.py とそのテストを調査してください。"
                "total() は合計金額に10を勝手に加算しており、discounted_total() は割引率を逆方向に適用しています。"
                "失敗原因を修正し、必要なら tests/test_order.py も更新して、最後に python -m pytest -q を再実行して全テスト成功を確認してください。"
            ),
            _seed_debug,
            (
                "src/order.py",
                "tests/test_order.py",
                "src/calculator.py",
                "tests/test_calculator.py",
            ),
            lambda r, rt, b: _capability_criteria(
                runtime=rt,
                root=r,
                baseline=b,
                allowed_paths={"src/order.py", "tests/test_order.py"},
                final_ok=(
                    _contains(r, "src/order.py", 'return sum(item["price"] for item in items)')
                    and _contains(r, "src/order.py", "1 - rate")
                ),
                require_pytest=True,
                require_failure_recovery=True,
            ),
        )
    )

    return tasks


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

    tasks = _build_tasks(root)

    print("J.A.R.V.I.S. Capability Matrix Benchmark V11")
    print(f"Model: {MODEL}")
    if thinking_mode:
        print(f"Thinking mode: {thinking_mode}")
    print(f"Workspace: {root}")
    print(f"Tasks: {len(tasks)}")
    print()

    results: list[dict[str, object]] = []
    capability_totals: dict[str, int] = {}
    capability_passed: dict[str, int] = {}
    passed = 0
    started_all = time.perf_counter()

    for index, task in enumerate(tasks, start=1):
        runtime.clear_session_context()
        if task.label == "Task 17: session context follow-up":
            # Session context is intentionally preserved within this task,
            # while the workspace remains task-isolated.
            pass

        _reset_workspace(root)
        task.seed(root)
        baseline = _workspace_snapshot(root)

        print(f"[RUN] {task.label}")
        try:
            result_text, elapsed, task_metrics = _run_once(runtime, task.prompt)
            prompt_count = 1
            follow_up_results: list[str] = []

            for follow_up in task.follow_up:
                follow_result, follow_elapsed, follow_metrics = _run_once(runtime, follow_up)
                prompt_count += 1
                elapsed += follow_elapsed
                follow_up_results.append(str(follow_result))
                for key, value in follow_metrics.items():
                    task_metrics[key] = int(task_metrics.get(key, 0)) + int(value)
                result_text = f"{result_text}\nFOLLOW-UP: {follow_result}"

            criteria = task.check(root, runtime, baseline)
            current_status = (
                runtime.current_task.state.status.value
                if runtime.current_task is not None
                else "unknown"
            )
            criteria["agent_completed"] = current_status == "completed"
            # first_attempt_clean is an independent quality signal. Recovery tasks
            # are allowed to have an initial failure, but the Agent itself must still
            # reach a completed state for the task to count as passed.
            required_criteria = {
                name: value
                for name, value in criteria.items()
                if name != "first_attempt_clean"
            }
            ok = all(required_criteria.values())
        except Exception as exc:
            result_text = f"{type(exc).__name__}: {exc}"
            elapsed = 0.0
            prompt_count = 0
            task_metrics = {}
            criteria = {
                "final_correctness": False,
                "scope_control": False,
                "safety": False,
                "verification": False,
                "failure_recovery": False,
                "tool_use": False,
                "first_attempt_clean": False,
                "agent_completed": False,
            }
            ok = False

        tool_counts = _tool_counts(runtime)
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {task.label} ({elapsed:.1f}s)")
        print("Criteria:", ", ".join(
            f"{name}={'PASS' if value else 'FAIL'}"
            for name, value in criteria.items()
        ))
        print(f"Final: {result_text}")
        print()

        passed += int(ok)
        results.append(
            {
                "index": index,
                "label": task.label,
                "capabilities": list(task.capabilities),
                "passed": ok,
                "elapsed_seconds": round(elapsed, 3),
                "prompt_count": prompt_count,
                "final": str(result_text),
                "criteria": criteria,
                "tool_counts": tool_counts,
                "metrics": task_metrics,
            }
        )

        for capability in task.capabilities:
            capability_totals[capability] = capability_totals.get(capability, 0) + 1
            capability_passed[capability] = (
                capability_passed.get(capability, 0) + int(ok)
            )

    capability_report = {
        capability: {
            "passed_tasks": capability_passed.get(capability, 0),
            "total_tasks": total,
            "pass_rate": round(capability_passed.get(capability, 0) / total, 3),
        }
        for capability, total in sorted(capability_totals.items())
    }

    first_attempt_clean_tasks = sum(
        int(result["criteria"].get("first_attempt_clean", False))
        for result in results
    )
    final_correct_tasks = sum(
        int(result["criteria"].get("final_correctness", False))
        for result in results
    )
    recovery_success_tasks = sum(
        int(result["criteria"].get("failure_recovery", False))
        for result in results
    )

    metrics = _trace_summary(trace_path)
    report = {
        "benchmark": "jarvis-capability-matrix-v11",
        "model": MODEL,
        "thinking_mode": thinking_mode,
        "max_iterations": max_iterations,
        "total_elapsed_seconds": round(time.perf_counter() - started_all, 3),
        "passed": passed,
        "total_tasks": len(tasks),
        "pass_rate": round(passed / len(tasks), 3),
        "summary": {
            "final_correctness_tasks": final_correct_tasks,
            "first_attempt_clean_tasks": first_attempt_clean_tasks,
            "failure_recovery_tasks": recovery_success_tasks,
        },
        "capabilities": capability_report,
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
        f"Metrics: LLM={metrics['llm_calls']} calls, Tool={metrics['tool_calls']} calls, "
        f"LLM={metrics['llm_duration_ms']}ms, Reasoning={metrics['reasoning_tokens']} tokens"
    )
    print(f"Result: {passed}/{len(tasks)} tasks passed ({passed / len(tasks):.1%})")
    print("Capability matrix:")
    for capability, data in capability_report.items():
        print(
            f"  {capability}: "
            f"{data['passed_tasks']}/{data['total_tasks']} "
            f"({data['pass_rate']:.0%})"
        )
    return 0 if passed == len(tasks) else 1


def main() -> int:
    args = _parse_args()
    if args.max_iterations < 1:
        raise SystemExit("--max-iterations must be at least 1")

    if args.keep_workspace:
        workspace = Path(tempfile.mkdtemp(prefix="jarvis-v11-benchmark-"))
        print(f"Benchmark workspace: {workspace}")
        return run_benchmark(
            workspace,
            model=args.model,
            max_iterations=args.max_iterations,
            thinking_mode=args.thinking_mode,
            output=args.output,
        )

    with tempfile.TemporaryDirectory(prefix="jarvis-v11-benchmark-") as temp_dir:
        return run_benchmark(
            Path(temp_dir),
            model=args.model,
            max_iterations=args.max_iterations,
            thinking_mode=args.thinking_mode,
            output=args.output,
        )


if __name__ == "__main__":
    raise SystemExit(main())
