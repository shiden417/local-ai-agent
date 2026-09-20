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
      ├─ edit_file
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
- edit_file - SEARCH / REPLACE方式の部分編集
- execute_command - PowerShellコマンド実行

## Tool design

LLMとの通信には、独自JSON文字列プロトコルではなく、LiteLLMのOpenAI互換形式によるNative Tool Callingを使用します。

ToolはToolRegistryに登録し、RuntimeはTool名から実装をディスパッチします。

ファイル系Toolは相対パスのみを受け付け、解決後のパスがAgent workspace外へ出ないことを確認します。シンボリックリンク等で解決後のパスがworkspace外になる場合も拒否します。

edit_fileは検索文字列がちょうど1回だけ一致する場合に変更し、変更結果としてdiffを返します。変更前にCLIでユーザー確認を行います。

execute_commandには30秒のデフォルトタイムアウト、タイムアウト時のプロセスツリー終了、LLMへ返す出力のサイズ上限があります。WindowsのPowerShell出力はUTF-8へ寄せて扱います。代表的な破壊・書き込みコマンドは実行前に確認します。

## Development

    python -m pytest -q

GitHub ActionsでもWindows Runner上でテストを実行します。

## Current status

現在は「Native Tool Calling + Tool Registry + 読み取り + 部分編集 + コマンド実行 + 基本的な自律ループ」までを実装しています。

まだ以下は未実装です。

- write_file
- より厳密なcommand sandbox / permission policy
- Git専用Tool
- Context compaction
- Repo Map
- Background / interactive process management
- 複数ターンにまたがる永続タスク状態

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
- 変更操作は確認可能にする
- 複雑な機能は基本ループが安定してから追加する
