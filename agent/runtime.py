from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from agent.context import ContextManager
from agent.llm import ask_llm
from agent.loop_guard import ToolLoopGuard
from agent.observation import truncate_text
from agent.safety import requires_confirmation
from agent.task import TaskState
from agent.task_manager import ManagedTask, TaskManager
from agent.tool_registry import ToolRegistry
from agent.tools import create_default_tool_registry


SYSTEM_PROMPT = """あなたはローカルで動作する汎用AI Agentです。
ユーザーの目的を達成するために、利用可能なツールを選択し、観測結果を確認しながら段階的に行動してください。

実行ルール:
- まずユーザーの目的を理解し、このTaskで必要な情報や操作を考えてください。
- Toolは「目的を達成するために必要なもの」だけを使用してください。
- 現在のworkspaceを調べる依頼では、まずlist_directoryで構造を確認し、既知のファイルはread_fileで内容を確認してください。
- search_filesは「ファイルの中にある特定の文字列・シンボルを探す」ためのToolです。ファイルの役割や「主要なファイル」のような自然言語カテゴリを検索語にしないでください。
- search_memoryは現在のworkspaceを見るためのToolではありません。過去の会話や保存済み情報が今回の目的に必要な場合だけ使用してください。
- save_memoryは今回だけの作業結果ではなく、将来のTaskでも役立つ情報を保存するときだけ使用してください。
- 既に取得した情報を同じToolで再取得しないでください。RuntimeはTask内の重複操作を検知して停止します。
- Tool結果に新しい情報がなければ、別の方法を考えるか、取得済み情報だけで回答を完成させてください。
- Toolが失敗したときは、エラーをそのまま繰り返さず原因を考えて別の方法を試してください。
- 変更や外部作用を伴うToolは、必要性を確認してから使用してください。
- ユーザーが求めていない変更を行わないでください。
- 作業が十分に完了したら、通常の文章で結果を説明してください。空のJSONや「{}」だけを最終回答にしないでください。

重要:
- あなたが判断し、RuntimeがToolを実行します。
- Task stateに含まれるRecent observationsは、これまでに得た事実です。重複した観測を無視して次の行動を選んでください。
"""


def _message_to_dict(message: Any) -> dict[str, Any]:
    if hasattr(message, "model_dump"):
        return message.model_dump(exclude_none=True)
    if isinstance(message, dict):
        return message
    return {
        "role": getattr(message, "role", "assistant"),
        "content": getattr(message, "content", None),
        "tool_calls": getattr(message, "tool_calls", None),
    }


def _tool_call_values(tool_call: Any) -> tuple[str, str, dict[str, Any]]:
    call_id = (
        tool_call.get("id", "")
        if isinstance(tool_call, dict)
        else getattr(tool_call, "id", "")
    )
    function = (
        tool_call.get("function", {})
        if isinstance(tool_call, dict)
        else getattr(tool_call, "function", None)
    )

    if isinstance(function, dict):
        name = function.get("name", "")
        arguments = function.get("arguments", {})
    else:
        name = getattr(function, "name", "")
        arguments = getattr(function, "arguments", {})

    if isinstance(arguments, str):
        arguments = json.loads(arguments)

    if not isinstance(arguments, dict):
        raise ValueError(f"Invalid tool arguments for {name}")

    return str(call_id), str(name), arguments


class AgentRuntime:
    def __init__(
        self,
        working_directory: str | Path,
        max_iterations: int = 10,
        tool_registry: ToolRegistry | None = None,
        confirm: Callable[[str], bool] | None = None,
        context_manager: ContextManager | None = None,
    ) -> None:
        self.working_directory = Path(working_directory).resolve()
        self.max_iterations = max_iterations
        self.tool_registry = tool_registry or create_default_tool_registry()
        self.confirm = confirm or self._default_confirm
        self.context_manager = context_manager or ContextManager()
        self.loop_guard = ToolLoopGuard()
        self.task_manager = TaskManager()
        self.current_task: ManagedTask | None = None
        self.task: TaskState | None = None

    @property
    def messages(self) -> list[dict[str, Any]]:
        """Return the current task history for compatibility and inspection."""
        if self.current_task is None:
            return [{"role": "system", "content": SYSTEM_PROMPT}]
        return self.current_task.messages

    @staticmethod
    def _default_confirm(message: str) -> bool:
        answer = input(f"\n{message}\nProceed? [y/N]: ")
        return answer.strip().lower() in {"y", "yes"}

    def run(self, user_input: str) -> str:
        current_task = self.task_manager.create(user_input)
        current_task.messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input},
        ]

        self.current_task = current_task
        self.task = current_task.state
        self.loop_guard.reset()
        self.task.start()
        self.task_manager.update_timestamp(current_task)

        for _ in range(self.max_iterations):
            self.task.begin_iteration()
            self.task_manager.update_timestamp(current_task)

            context_messages = self.context_manager.prepare(current_task.messages)
            llm_messages = [
                *context_messages,
                {
                    "role": "system",
                    "content": (
                        "Current task execution state. Treat Recent observations "
                        "as already-known information.\n"
                        f"{self.task.snapshot()}\n"
                        "Choose the smallest next action that advances the goal."
                    ),
                },
            ]

            response = ask_llm(
                llm_messages,
                tools=self.tool_registry.schemas_for(self.task.goal),
            )
            message = response.choices[0].message
            tool_calls = getattr(message, "tool_calls", None) or []
            content = getattr(message, "content", None) or ""

            if not tool_calls:
                if self._is_invalid_final_response(content, current_task.messages):
                    current_task.messages.append(_message_to_dict(message))
                    current_task.messages.append(
                        {
                            "role": "system",
                            "content": (
                                "The previous response was empty or only an "
                                "empty JSON container. Continue the task using "
                                "the available observations and provide a "
                                "direct answer to the user's request."
                            ),
                        }
                    )
                    continue

                current_task.messages.append(_message_to_dict(message))
                self.task.complete()
                self.task_manager.update_timestamp(current_task)
                return content

            current_task.messages.append(_message_to_dict(message))

            for tool_call in tool_calls:
                try:
                    call_id, name, arguments = _tool_call_values(tool_call)
                except (TypeError, ValueError, json.JSONDecodeError) as exc:
                    print(f"[Tool Error] {exc}")
                    self.task.record_tool(
                        "invalid_tool_call",
                        succeeded=False,
                        summary=str(exc),
                        signature=f"invalid_tool_call:{type(exc).__name__}:{exc}",
                    )
                    current_task.messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": "",
                            "content": json.dumps(
                                {"ok": False, "error": str(exc)},
                                ensure_ascii=False,
                            ),
                        }
                    )
                    continue

                call_count = self.loop_guard.record(name, arguments)
                if self.loop_guard.is_repetition(name, arguments):
                    result = {
                        "ok": False,
                        "error": self.loop_guard.message(name, arguments),
                        "repeated_tool_call": True,
                        "call_count": call_count,
                    }
                    print("[Tool] repeated call blocked")
                elif requires_confirmation(name, arguments, self.tool_registry):
                    summary = self._confirmation_message(name, arguments)
                    if not self.confirm(summary):
                        result = {
                            "ok": False,
                            "error": "User rejected the operation.",
                            "user_rejected": True,
                        }
                        print("[Tool] rejected by user")
                    else:
                        result = self._execute_tool(name, arguments)
                else:
                    result = self._execute_tool(name, arguments)

                serialized = json.dumps(result, ensure_ascii=False, sort_keys=True)
                bounded, truncated = truncate_text(serialized)

                observation_summary = self._observation_summary(result)
                result_signature = json.dumps(
                    result,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                signature = (
                    f"{name}:"
                    + hashlib.sha256(
                        result_signature.encode("utf-8")
                    ).hexdigest()
                )
                self.task.record_tool(
                    name,
                    succeeded=bool(result.get("ok")),
                    summary=observation_summary,
                    signature=signature,
                )
                self.task_manager.update_timestamp(current_task)

                print("[Result]")
                print(bounded)
                if truncated:
                    print("[Result] output truncated before returning to the model.")

                current_task.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": name,
                        "content": bounded,
                    }
                )

        self.task.hit_max_iterations()
        self.task_manager.update_timestamp(current_task)
        return "Agentの最大反復回数に達したため、処理を終了しました。"

    @staticmethod
    def _is_invalid_final_response(
        content: str,
        messages: list[dict[str, Any]] | None = None,
    ) -> bool:
        normalized = content.strip()
        if not normalized or normalized in {"{}", "[]"}:
            return True

        if messages:
            return any(
                message.get("role") == "tool"
                and str(message.get("content", "")).strip() == normalized
                for message in messages
            )

        return False

    @staticmethod
    def _observation_summary(result: dict[str, Any]) -> str:
        """Create a small deterministic summary for task state."""
        summary = json.dumps(
            result,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        summary, _ = truncate_text(summary, 900)
        return summary

    def list_tasks(self) -> list[ManagedTask]:
        return self.task_manager.list_tasks()

    def _execute_tool(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        print(f"\n[Tool] {name}")
        print(f"[Working Directory] {self.working_directory}")
        print(f"[Arguments] {json.dumps(arguments, ensure_ascii=False)}")

        return self.tool_registry.execute(
            name,
            arguments,
            self.working_directory,
        )

    @staticmethod
    def _confirmation_message(
        name: str,
        arguments: dict[str, Any],
    ) -> str:
        if name == "edit_file":
            return (
                "Agentがローカルファイルを変更しようとしています.\n"
                f"path: {arguments.get('path', '')}"
            )

        return (
            "Agentが確認の必要な操作を実行しようとしています。\n"
            f"tool: {name}\n"
            f"arguments: {json.dumps(arguments, ensure_ascii=False)}"
        )
