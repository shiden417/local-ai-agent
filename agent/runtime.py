from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from agent.llm import ask_llm
from agent.observation import truncate_text
from agent.safety import requires_confirmation
from agent.task import TaskState
from agent.tool_registry import ToolRegistry
from agent.tools import create_default_tool_registry


SYSTEM_PROMPT = """あなたはローカルで動作する汎用AI Agentです。
ユーザーの目的を達成するために、利用可能なツールを適切に組み合わせて自律的に行動してください。

重要なルール:
- 依頼の目的を理解してから行動してください。
- 複雑な依頼では、実行前に達成までの手順を内部で小さく分解してください。
- 必要な情報を調査し、観測結果を確認してから次の行動を判断してください。
- 不可逆な変更や確認が必要な操作を急いで実行しないでください。
- ツールを使った結果に基づいて、必要なら追加のツールを呼び出してください。
- ツールが失敗した場合は、エラー内容を分析して別の方法を検討してください。
- 1回の判断では必要最小限の操作を選んでください。
- 現在の作業環境で利用できる範囲を超えてアクセスしようとしないでください。
- ユーザーの確認が必要な操作は、確認が得られてから実行してください。
- 作業が完了したら、結果と重要な変更点を通常の文章で説明してください。

利用可能なツールは、その時点でRuntimeから提供されます。
各Toolのdescriptionとparametersを読み、目的に最も適したToolを選択してください。
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
    ) -> None:
        self.working_directory = Path(working_directory).resolve()
        self.max_iterations = max_iterations
        self.tool_registry = tool_registry or create_default_tool_registry()
        self.confirm = confirm or self._default_confirm
        self.task: TaskState | None = None
        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

    @staticmethod
    def _default_confirm(message: str) -> bool:
        answer = input(f"\n{message}\nProceed? [y/N]: ")
        return answer.strip().lower() in {"y", "yes"}

    def run(self, user_input: str) -> str:
        self.task = TaskState(goal=user_input)
        self.task.start()
        self.messages.append({"role": "user", "content": user_input})

        for _ in range(self.max_iterations):
            self.task.begin_iteration()

            llm_messages = [
                *self.messages,
                {
                    "role": "system",
                    "content": (
                        "Current task state: "
                        f"{self.task.snapshot()}\n"
                        "Use this only as execution state. Do not expose internal "
                        "task-state details unless the user asks."
                    ),
                },
            ]

            response = ask_llm(
                llm_messages,
                tools=self.tool_registry.schemas,
            )
            message = response.choices[0].message
            tool_calls = getattr(message, "tool_calls", None) or []
            content = getattr(message, "content", None) or ""

            if not tool_calls:
                self.messages.append(_message_to_dict(message))
                self.task.complete()
                return content

            self.messages.append(_message_to_dict(message))

            for tool_call in tool_calls:
                try:
                    call_id, name, arguments = _tool_call_values(tool_call)
                except (TypeError, ValueError, json.JSONDecodeError) as exc:
                    print(f"[Tool Error] {exc}")
                    self.task.fail(f"Invalid tool call: {exc}")
                    self.messages.append(
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

                if requires_confirmation(name, arguments, self.tool_registry):
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

                self.task.record_tool(name, succeeded=bool(result.get("ok")))

                serialized = json.dumps(result, ensure_ascii=False, indent=2)
                bounded, truncated = truncate_text(serialized)

                print("[Result]")
                print(bounded)
                if truncated:
                    print("[Result] output truncated before returning to the model.")

                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": name,
                        "content": bounded,
                    }
                )

        self.task.hit_max_iterations()
        return "Agentの最大反復回数に達したため、処理を終了しました。"

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
