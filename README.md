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

Agent Coreには、1つの依頼を独立して追跡するTaskStateと軽量な実行状態・観測履歴を実装しています。別Planner Agentを増やさず、Qwen3:8Bへの呼び出し回数を必要以上に増やさない方針です。

Long-term MemoryはAgent CoreのTask履歴とは分離し、ユーザーホーム配下のローカルJSONへ永続化します。検索は現在キーワードベースで、外部サービスやクラウドへ送信しません。

- list_directory - workspace内の一覧取得
- read_file - テキストファイルの読み取り
- search_files - ローカルファイル検索
- edit_file - SEARCH / REPLACE方式の部分編集
- execute_command - PowerShellコマンド実行
- save_memory - 将来も利用する情報をローカルMemoryへ保存
- search_memory - 過去のローカルMemoryを検索

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
- Agent loopの収束・安定化
- Taskごとの独立した会話履歴
- Task observation ledger（観測結果・新規情報・進捗）
- Task内の重複Tool Call検知・Tool quarantine
- Context compaction
- Task Manager
- Task-aware Tool Capability Routing
- permission policyの強化

### Tools
- Git
- Web / HTTP
- Database
- Windows automation
- Scheduler
- Notifications

### Interaction
- 会話履歴
- Taskの一覧確認
- 音声入力 (STT)
- 音声出力 (TTS)
- GUI
- 常駐 / event-driven execution

### Intelligence
- Goal / task decomposition
- Long-term memoryの高度化
- Planningの強化
- Proactive behavior
- 長期的な自己改善・評価基盤


## Design principles

- LLMは判断する
- Runtimeは実行する
- Toolは明確な責務を持つ
- Agent Coreに特定用途のロジックを埋め込まない
- ToolはRegistry経由で追加できるようにする
- Tool結果はLLMへ返す前にサイズを制限する
- Task内容に応じて必要なTool capabilityだけをLLMへ公開する
- 同じTool + 同じ引数のTask内重複実行を検知し、探索ループを抑止する
- Tool結果をTask observationとして保持し、新しい情報が得られたかを追跡する
- Task間の会話履歴を混在させない
- ContextはTool CallとTool Resultの会話ブロックを壊さずに圧縮する
- workspace外のアクセスを許可しない
- 変更操作は確認可能にする
- 巨大なAgent Frameworkをそのまま導入せず、必要な機能を段階的に自作する