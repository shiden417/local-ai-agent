# local-ai-agent

Ollama + Qwen3:8B + LiteLLM を使ったローカルAI Agentです。

## Architecture

    User
      ↓
    Agent Runtime
      ↓
    LiteLLM
      ↓
    Ollama / Qwen3:8B
      ↓
    Native Tool Calling
      ↓
    Tool Registry / Dispatcher
      ├─ list_directory
      ├─ read_file
      ├─ search_files
      └─ execute_command
      ↓
    Observation
      ↓
    Qwen3 decides next step
      ↓
    Retry / Continue
      ↓
    Final answer

## Requirements

- Windows
- Python 3.12+
- Ollama
- Qwen3:8B
- LiteLLM

## Setup

    python -m venv .venv
    .\\.venv\\Scripts\\Activate.ps1
    pip install -r requirements.txt

## Run

Agentを操作したい作業ディレクトリで起動します。

    python agent.py

Agent Runtimeは起動時のカレントディレクトリを作業ディレクトリとして固定します。

## Current tools

- list_directory - ワークスペース内の一覧取得
- read_file - テキストファイルの読み取り
- search_files - テキスト検索
- execute_command - PowerShellコマンド実行

## Tool design

LLMとの通信には、独自JSON文字列プロトコルではなく、LiteLLMのOpenAI互換形式によるNative Tool Callingを使用します。

ToolはToolRegistryに登録し、RuntimeはTool名から実装をディスパッチします。

ファイル系Toolは相対パスのみを受け付け、解決後のパスがAgent workspace外へ出ないことを確認します。シンボリックリンク等で解決後のパスがworkspace外になる場合も拒否します。

execute_commandには30秒のデフォルトタイムアウトと、LLMへ返す出力のサイズ上限があります。WindowsのPowerShell出力はUTF-8へ寄せて扱います。

## Development

    pytest

## Current status

現在は次の基盤まで実装しています。

- Native Tool Calling
- Tool Registry / Dispatcher
- list_directory
- read_file
- search_files
- execute_command
- Agent Loop
- Tool observationの出力制限
- workspace path traversal対策
- command timeout

まだ以下は未実装です。

- edit_file
- write_file
- Safety / confirmation
- Git専用Tool
- Context compaction
- Repo Map
- Background / interactive process management

## Roadmap

1. Native Tool Calling
2. Tool Registry / Dispatcher
3. 読み取り系Tool
4. Agent Loop / error recovery
5. edit_file (SEARCH / REPLACE)
6. Safety / confirmation
7. Git integration
8. Context management / Repo Map
9. UX improvements

## Design principles

小規模なローカルLLM向けに、巨大なAgent Frameworkをそのまま導入せず、必要な機能を段階的に自作します。

- LLMは判断する
- Runtimeは実行する
- Toolは明確な責務を持つ
- Tool結果はLLMへ返す前にサイズを制限する
- workspace外のファイルへアクセスさせない
- 複雑な機能は基本ループが安定してから追加する
