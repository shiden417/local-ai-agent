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
    Conversation Context
      ↓
    Agent Runtime
      ↓
    Capability Router
      ├─ Direct: obvious conversation → no tools
      ├─ Scoped: expose relevant capability families
      └─ Open: ambiguous → no on-demand tools; avoid speculative actions
      ↓
    LLM abstraction
      ↓
    Ollama / Qwen3:8B
      ↓
    Native Tool Calling
      ↓
    Tool Registry / Dispatcher
      ├─ Built-in Tools
      ├─ Temporary Script capability
      ├─ Capability Management
      │    └─ Quarantine → Promote → Load
      └─ future capabilities
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

Agent Coreには、1つの依頼を独立して追跡するTaskStateと軽量な実行状態・観測履歴を実装しています。さらに、Action identity（同一Tool呼び出し）、Observation identity（得られた知識）、Progress（目的への前進）を分離して管理します。Runtimeは毎回Task dashboardをLLMへ提示し、Capability RouterでToolの公開範囲だけを調整します。最終的な「Toolを使うか」「どのToolを使うか」はQwen3:8Bが判断します。明らかな会話はDirect、具体的な作業はScoped、曖昧な依頼はOpenとして扱います。Openでは現在、推測によるTool実行を防ぐためon-demand Toolを公開しません。将来、Web/APIなどの追加Capabilityを導入する際に、Openの扱いを拡張できる構造にします。別Planner Agentを増やさず、Qwen3:8Bへの呼び出し回数を必要以上に増やさない方針です。

Long-term MemoryはAgent CoreのTask履歴とは分離し、ユーザーホーム配下のローカルJSONへ永続化します。検索は現在キーワードベースで、外部サービスやクラウドへ送信しません。

- list_directory - workspace内の一覧取得
- read_file - テキストファイルの読み取り
- search_files - ローカルファイル検索
- file_mutation - ローカルファイルの作成・編集・削除
- execute_command - PowerShellコマンド実行
- run_python_script - 専用Toolがない処理を一時Python Scriptとして実行
- list_promotion_candidates - 繰り返し成功したRecipeをPromotion候補として取得
- generate_plugin - RecipeからPlugin候補をLLM生成
- test_plugin_candidate - 生成Pluginを子プロセスで確認付き検証
- stage_plugin - 新しいPluginを検疫領域へ配置
- promote_plugin - 検疫済みPluginを確認付きで有効化
- save_memory - 将来も利用する情報をローカルMemoryへ保存
- search_memory - 過去のローカルMemoryを検索

成功したrun_python_scriptはRecipeStoreへ自動保存されます。関連する次のTaskでは、過去に成功したRecipeをLLMへ参考情報として提示します。Recipeは成功実績の再利用を目的としたもので、自動で正式Pluginにはしません。Capability管理Taskでは、候補を `generate_plugin` でPlugin化し、`test_plugin_candidate` で実行検証した後、`stage_plugin` → `promote_plugin` の順で永続化できます。

永続Capabilityを作る場合は、Agentがstage_pluginでPluginを検疫領域へ配置し、構文・契約を検証した後、promote_pluginでユーザー確認を経て有効化できます。有効Pluginは固定ブートストラップ経由の子Pythonプロセスとして実行され、Agent Coreのプロセス内ではPluginコードを実行しません。

LLMとの通信には、独自JSON文字列プロトコルではなく、LiteLLMのNative Tool Calling形式を使用します。

ToolはToolRegistryに登録され、RuntimeはTool名から実装をディスパッチします。

## Safety

Agentの操作には実行環境に応じた安全策を設定します。

- 作業ディレクトリ内の相対パスはworkspace外へ脱出できないよう制限
- ユーザーが明示したローカル絶対パスはFile Toolで扱える
- Toolごとに確認が必要か設定可能
- file_mutationによる変更は実行前にユーザー確認
- 代表的な破壊・書き込み系PowerShell/Git操作は確認
- execute_commandは30秒timeout
- run_python_scriptは15秒timeout（最大30秒）・スクリプト12,000文字・出力8,000文字に制限
- run_python_scriptは子プロセスで実行し、実行前にユーザー確認
- Pluginは有効化後も常にユーザー確認が必要
- Plugin実行は子プロセスで行い、Coreプロセス内ではPluginコードを実行しない
- timeout時はプロセスを終了
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
- Taskごとの独立した実行履歴
- セッション内の会話コンテキスト
- Task observation ledger（観測結果・新規情報・進捗）
- Runtime-managed execution dashboard
- Action / Observation / Progressの分離
- Direct / Scoped / OpenのCapability routing
- Task内の重複Tool Call検知・Tool quarantine
- Tool Failure Recoveryの一時Quarantine
- Context compaction
- Task Manager
- Task-aware Capability routing
- Direct / Scoped / Open tool exposure
- permission policyの強化

### Capability learning
- 成功した一時ScriptのRecipe化
- Recipe再利用
- Recipe使用回数に基づくPromotion候補検出
- RecipeからPlugin候補をLLM生成
- 生成Pluginの構文・契約検証
- 生成Pluginの子プロセスによる確認付きテスト
- PluginをQuarantineへStage
- 確認付きPromotionと動的ロード
- Agentからのlist_promotion_candidates / stage_plugin / promote_pluginによるCapability獲得

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
- Task内容に応じてTool capabilityの公開範囲だけを調整する
- 明らかな会話ではToolを公開せず、具体的な作業では該当Capabilityを公開し、曖昧な依頼ではQwen3:8Bへ判断を委ねる
- 同じTool + 同じ引数のTask内重複実行を検知し、該当ToolをTask単位で一時無効化する
- Tool結果の意味的な観測同一性を判定し、新しい情報が得られたかを追跡する
- 観測の新規性と、目的に対する実際のProgressを別々に判定する
- Runtime-managed dashboardでGoal / Phase / Progress / Recent observations / Disabled toolsをLLMへ明示する
- Task間の実行履歴を混在させない
- セッション会話はTask履歴とは別に保持し、直近の会話だけをLLMへ渡す
- ContextはTool CallとTool Resultの会話ブロックを壊さずに圧縮する
- workspace外のアクセスを許可しない
- 変更操作は確認可能にする
- 巨大なAgent Frameworkをそのまま導入せず、必要な機能を段階的に自作する