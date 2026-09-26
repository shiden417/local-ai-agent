from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_PATH = PROJECT_ROOT / "finetuning" / "data" / "agent_sft.jsonl"

TRAINING_SYSTEM_PROMPT = (
    "あなたはNEXAというローカルAI Agentです。"
    "ユーザーの目的を達成するため、必要なToolだけを選び、"
    "観測結果を確認しながら最小限の操作で進めます。"
    "失敗したら原因を確認して別の手段で復旧します。"
    "変更した場合は結果を検証し、未実行の操作を完了したとは言いません。"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate successful NEXA Agent trajectories for QLoRA/SFT."
    )
    parser.add_argument("--model", default="qwen/qwen3-8b")
    parser.add_argument(
        "--thinking-mode",
        choices=("default", "think", "no_think"),
        default="default",
    )
    parser.add_argument("--max-iterations", type=int, default=12)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return parser.parse_args()


def _message_dict(message: Any) -> dict[str, Any]:
    if hasattr(message, "model_dump"):
        return message.model_dump(exclude_none=True)
    if isinstance(message, dict):
        return dict(message)
    result: dict[str, Any] = {}
    for key in ("role", "content", "name", "tool_call_id", "tool_calls"):
        value = getattr(message, key, None)
        if value is not None:
            result[key] = value
    return result


def _sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize(v) for v in value]
    if isinstance(value, str):
        value = value.replace(str(PROJECT_ROOT), "<PROJECT_ROOT>")
        value = value.replace(str(Path.home()), "<HOME>")
        return value
    return value


def _tool_names(messages: list[dict[str, Any]]) -> set[str]:
    names: set[str] = set()
    for message in messages:
        if message.get("role") == "tool" and message.get("name"):
            names.add(str(message["name"]))
        for call in message.get("tool_calls") or []:
            function = call.get("function", {}) if isinstance(call, dict) else {}
            if isinstance(function, dict) and function.get("name"):
                names.add(str(function["name"]))
    return names


def _messages_for_training(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = [
        {"role": "system", "content": TRAINING_SYSTEM_PROMPT}
    ]

    for raw in messages:
        message = _message_dict(raw)
        role = message.get("role")

        if role == "system":
            continue

        if role == "assistant":
            message.pop("reasoning_content", None)
            message.pop("reasoning", None)
            message.pop("thinking", None)

        if role in {"user", "assistant", "tool"}:
            result.append(_sanitize(message))

    return result


def _tool_schemas(runtime: Any, names: set[str]) -> list[dict[str, Any]]:
    schemas: list[dict[str, Any]] = []
    for name in sorted(names):
        tool = runtime.tool_registry.get(name)
        if tool is not None:
            schemas.append(_sanitize(tool.schema()))
    return schemas


def main() -> int:
    args = parse_args()

    os.environ["LM_STUDIO_MODEL"] = args.model
    os.environ["LM_STUDIO_THINKING_MODE"] = args.thinking_mode

    from agent.memory import MemoryStore
    from agent.runtime import AgentRuntime
    from agent.session import SessionManager
    from agent.tools import create_default_tool_registry
    from agent.trace import TraceRecorder
    from tools.benchmark_agent_v11 import _build_tasks, _reset_workspace, _snapshot

    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="nexa-sft-") as temp_dir:
        root = Path(temp_dir)
        runtime = AgentRuntime(
            working_directory=root,
            max_iterations=args.max_iterations,
            tool_registry=create_default_tool_registry(
                memory_store=MemoryStore(root / "memory.json")
            ),
            session_manager=SessionManager(),
            confirm=lambda _message: True,
            trace_recorder=TraceRecorder(root / "trace.jsonl"),
        )

        tasks = _build_tasks(root)
        written = 0

        with output_path.open("w", encoding="utf-8") as handle:
            for task in tasks:
                runtime.clear_session_context()
                _reset_workspace(root)
                task.seed(root)
                baseline = _snapshot(root, task.expected_paths)

                print(f"[RUN] {task.label}")

                try:
                    runtime.run(task.prompt)
                    criteria = task.check(root, runtime, baseline)
                    ok = all(criteria.values())
                except Exception as exc:
                    criteria = {"exception": False}
                    ok = False
                    print(f"  exception: {type(exc).__name__}: {exc}")

                print(f"[{'PASS' if ok else 'FAIL'}] {task.label}")

                if not ok or runtime.current_task is None:
                    continue

                messages = _messages_for_training(runtime.current_task.messages)
                tool_names = _tool_names(messages)

                if len(messages) < 3:
                    continue

                sample = {
                    "messages": messages,
                    "tools": _tool_schemas(runtime, tool_names),
                    "metadata": {
                        "source": "benchmark_agent_v11",
                        "task": task.label,
                        "criteria": criteria,
                    },
                }

                handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
                written += 1
                print("  exported: " + (",".join(sorted(tool_names)) or "none"))

        print()
        print(f"Training samples written: {written}")
        print(f"Dataset: {output_path}")
        return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
