import json
from pathlib import Path

from agent.llm import ask_llm
from agent.parser import parse_tool_call
from tools.execute_command import execute_command


SYSTEM_PROMPT = """あなたはローカルAI Agentです。
ユーザーの依頼を達成するために、必要ならツールを使用してください。

現在使用できるツール:

execute_command
- Agentの作業ディレクトリ内でPowerShellコマンドを実行します。
- 作業ディレクトリはAgent Runtimeが固定しているため、コマンド内で勝手に別の場所へ移動しないでください。
- ツールを使用する場合は、必ず次のJSONだけを出力してください。

{
  "name": "execute_command",
  "arguments": {
    "command": "PowerShellコマンド"
  }
}

ツールを使用しない場合は、通常の文章で回答してください。

ツール実行結果を受け取った場合は、その結果を分析してください。
必要なら追加のツールを使用し、作業が完了するまで継続してください。
作業が完了したら、ユーザーに結果を通常の文章で説明してください。
"""


class AgentRuntime:
    def __init__(
        self,
        working_directory: str | Path,
        max_iterations: int = 10,
    ) -> None:
        self.working_directory = Path(working_directory).resolve()
        self.max_iterations = max_iterations
        self.messages: list[dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

    def run(self, user_input: str) -> str:
        self.messages.append({"role": "user", "content": user_input})

        for _ in range(self.max_iterations):
            response = ask_llm(self.messages)
            tool_call = parse_tool_call(response)

            if tool_call is None:
                self.messages.append({"role": "assistant", "content": response})
                return response

            command = tool_call["command"]
            print("\n[Tool] execute_command")
            print(f"[Working Directory] {self.working_directory}")
            print(f"[Command] {command}")

            tool_result = execute_command(
                command,
                working_directory=self.working_directory,
            )

            print(f"[Exit Code] {tool_result['exit_code']}")
            if tool_result["stdout"]:
                print("[stdout]")
                print(tool_result["stdout"])
            if tool_result["stderr"]:
                print("[stderr]")
                print(tool_result["stderr"])

            self.messages.append({"role": "assistant", "content": response})
            self.messages.append({
                "role": "user",
                "content": (
                    "ツール実行結果:\n"
                    + json.dumps(tool_result, ensure_ascii=False, indent=2)
                ),
            })

        return "Agentの最大反復回数に達したため、処理を終了しました。"
