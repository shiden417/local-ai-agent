import json
import subprocess

from litellm import completion


MODEL = "ollama/qwen3:8b"


def ask_llm(messages):
    response = completion(
        model=MODEL,
        messages=messages,
    )
    return response.choices[0].message.content


def parse_tool_call(text):
    """LLMの出力からexecute_commandのJSONを取得する。"""

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None

    if data.get("name") != "execute_command":
        return None

    arguments = data.get("arguments", {})
    command = arguments.get("command")

    if not command:
        return None

    return {
        "name": "execute_command",
        "command": command,
    }


def execute_command(command):
    """PowerShellコマンドを実行する。"""

    try:
        result = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                command,
            ],
            capture_output=True,
            text=True,
            encoding="cp932",
            errors="replace",
            timeout=30,
        )

        return {
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

    except subprocess.TimeoutExpired:
        return {
            "exit_code": -1,
            "stdout": "",
            "stderr": "コマンドが30秒以内に終了しませんでした。",
        }

    except Exception as e:
        return {
            "exit_code": -1,
            "stdout": "",
            "stderr": f"コマンド実行中にエラーが発生しました: {e}",
        }


def main():
    messages = [
        {
            "role": "system",
            "content": """
あなたはローカルAI Agentです。

ユーザーの依頼を達成するために、必要ならツールを使用してください。

現在使用できるツール:

execute_command
- PowerShellコマンドを実行します。
- ツールを使用する場合は、必ず次のJSONだけを出力してください。

{
  "name": "execute_command",
  "arguments": {
    "command": "PowerShellコマンド"
  }
}

ツールを使用しない場合は、通常の文章で回答してください。

ツール実行結果を受け取った場合は、その結果を分析して、
必要なら次のツールを使用してください。
作業が完了したら、ユーザーに結果を通常の文章で説明してください。
""",
        }
    ]

    while True:
        user_input = input("> ")

        if user_input.lower() in ("exit", "quit"):
            break

        messages.append(
            {
                "role": "user",
                "content": user_input,
            }
        )

        while True:
            result = ask_llm(messages)

            tool_call = parse_tool_call(result)

            if tool_call is None:
                print(result)

                messages.append(
                    {
                        "role": "assistant",
                        "content": result,
                    }
                )

                break

            command = tool_call["command"]

            print(f"\n[Tool] execute_command")
            print(f"[Command] {command}")

            tool_result = execute_command(command)

            print(f"[Exit Code] {tool_result['exit_code']}")

            if tool_result["stdout"]:
                print("[stdout]")
                print(tool_result["stdout"])

            if tool_result["stderr"]:
                print("[stderr]")
                print(tool_result["stderr"])

            messages.append(
                {
                    "role": "assistant",
                    "content": result,
                }
            )

            messages.append(
                {
                    "role": "user",
                    "content": (
                        "ツール実行結果:\n"
                        + json.dumps(
                            tool_result,
                            ensure_ascii=False,
                            indent=2,
                        )
                    ),
                }
            )


if __name__ == "__main__":
    main()