from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent.llm import ask_llm
from agent.observation import truncate_text
from agent.tool_registry import ToolRegistry
from agent.tools import create_default_tool_registry


SYSTEM_PROMPT = """あなたはローカルAI Agentです。
ユーザーの依頼を達成するために、必要なツールを自律的に使用してください。

重要なルール:
- ツールを使用する前に、必要な情報を調査してください。
- ファイルを変更する前に、対象ファイルを読み、周辺コードを理解してください。
- 1回の判断では、必要最小限のツールを使用してください。
- ツール実行結果を確認し、必要なら次のツールを呼び出してください。
- コマンドが失敗した場合は、エラー内容を分析して別の方法を試してください。
- Agentの作業ディレクトリの外へアクセスしようとしないでください。
- 作業が完了したら、最終結果を通常の文章で説明してください。

現在使用できるツール:
- list_directory: 作業ディレクトリ内のファイル・ディレクトリ一覧
- read_file: 作業ディレクトリ内のテキストファイルの読み取り
- search_files: 作業ディレクトリ内の文字列検索
- execute_command: 作業ディレクトリをカレントディレクトリとしてPowerShellを実行

ファイル操作では、できるだけ専用Toolを優先してください。
execute_commandはビルド、テスト、Git確認など、専用Toolがない操作に使用してください。
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
    ) -> None:
        self.working_directory = Path(working_directory).resolve()
        self.max_iterations = max_iterations
        self.tool_registry = tool_registry or create_default_tool_registry()
        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

    def run(self, user_input: str) -> str:
        self.messages.append({"role": "user", "content": user_input})

        for _ in range(self.max_iterations):
            response = ask_llm(
                self.messages,
                tools=self.tool_registry.schemas,
            )
            message = response.choices[0].message
            tool_calls = getattr(message, "tool_calls", None) or []
            content = getattr(message, "content", None) or ""

            if not tool_calls:
                self.messages.append(_message_to_dict(message))
                return content

            self.messages.append(_message_to_dict(message))

            for tool_call in tool_calls:
                try:
                    call_id, name, arguments = _tool_call_values(tool_call)
                except (TypeError, ValueError, json.JSONDecodeError) as exc:
                    print(f"[Tool Error] {exc}")
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

                print(f"\n[Tool] {name}")
                print(f"[Working Directory] {self.working_directory}")
                print(f"[Arguments] {json.dumps(arguments, ensure_ascii=False)}")

                result = self.tool_registry.execute(
                    name,
                    arguments,
                    self.working_directory,
                )

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

        return "Agentの最大反復回数に達したため、処理を終了しました。"
