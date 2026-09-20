# local-ai-agent

Ollama + Qwen3:8B + LiteLLM を使ったローカルAI Agentです。

## Goal

最初の実用ユースケースはソフトウェア開発支援ですが、最終的にはJ.A.R.V.I.S.のような汎用Local AI Agentへ発展させることを目標にします。

Agent Coreは特定用途に依存せず、Toolを追加することで能力を拡張できる構成を採用します。

想定する将来の能力:

- ローカルファイル・PC操作
- ソフトウェア開発
- Web / API
- データベース
- Git
- スケジュール・自動化
- 音声入出力
- Memory / Task management

## Architecture

    User
      ↓
    Agent Runtime
      ↓
    LLM abstraction
      ↓
    Ollama / Qwen3:8B
      ↓
    Native Tool Calling
      ↓
    Tool Registry / Dispatcher
      ├─ Local File tools
      ├─ Process / OS tools
      ├─ Coding tools
      └─ future tools
      ↓
    Observation / Context Management
      ↓
    Agent decides next action
      ↓
    Retry / Continue
      ↓
    Final answer

## Current implementation

現時点では、ローカルPC上でのCoding / Automationを最初の用途として、次のToolを提供しています。

Agent Coreには、1つの依頼を追跡するTaskStateと軽量なImplicit Planningを実装しています。別Planner Agentを増やさず、Qwen3:8Bへの呼び出し回数を増やさない方針です。

- list_directory - workspace内の一覧取得
- read_file - テキストファイルの読み取り
- search_files - ローカルファイル検索
- edit_file - SEARCH / REPLACE方式の部分編集
- execute_command - PowerShellコマンド実行

LLMとの通信には、独自JSON文字列プロトコルではなく、LiteLLMのNative Tool Calling形式を使用します。

ToolはToolRegistryに登録され、RuntimeはTool名から実装をディスパッチします。

## Safety

Agentの操作には実行環境に応じた安全策を設定します。

- workspace外へのファイルアクセスを拒否
- Toolごとに確認が必要か設定可能
- edit_fileは変更前にユーザー確認
- 代表的な破壊・書き込み系PowerShell/Git操作は確認
- execute_commandは30秒timeout
- timeout時はPowerShellプロセスツリーを終了
- Tool結果のサイズを制限してLLMへ返す

Safety判定は現在は保守的なヒューリスティックであり、完全なセキュリティサンドボックスではありません。

## Requirements

- Windows
- Python 3.12+
- Ollama
- Qwen3:8B
- LiteLLM

## Setup

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    pip install -r requirements.txt

## Run

Agentを操作したい作業ディレクトリで起動します。

    python agent.py

Agent Runtimeは起動時のカレントディレクトリをworkspaceとして固定します。

## Development

    python -m pytest -q

GitHub ActionsでもWindows Runner上でテストを実行します。

## Roadmap

### Agent Core

- Agent loopの安定化
- Task state（1タスクの状態・反復・Tool履歴）
- Context compaction（実装済みの基礎）
- 複数タスクのtask management
- Memory
- より明確なpermission policy

### Tools

- Git
- Web / HTTP
- Database
- Windows automation
- Scheduler
- Notifications

### Interaction

- 会話履歴
- 音声入力 (STT)
- 音声出力 (TTS)
- GUI
- 常駐 / event-driven execution

### Intelligence

- Implicit planning（LLM内で計画し、Runtimeが状態を追跡）
- Planning
- Long-term memory
- Proactive behavior
- Goal / task decomposition

## Design principles

- LLMは判断する
- Runtimeは実行する
- Toolは明確な責務を持つ
- Agent Coreに特定用途のロジックを埋め込まない
- ToolはRegistry経由で追加できるようにする
- Tool結果はLLMへ返す前にサイズを制限する
- ContextはTool CallとTool Resultの会話ブロックを壊さずに圧縮する
- workspace外のアクセスを許可しない
- 変更操作は確認可能にする
- 巨大なAgent Frameworkをそのまま導入せず、必要な機能を段階的に自作する