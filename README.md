# local-ai-agent

LM Studio + ローカルLLMを基盤にした、無料・ローカル・無制限利用を目標とする汎用AI Agentです。

現在の既定モデルは **Gemma 4 E4B QAT** です。モデルは環境変数で変更できます。

## Goal

最終目標は、Iron ManのJ.A.R.V.I.S.のように、文字で指示するとPC上で状況を判断し、必要なToolを使い、結果を確認しながら仕事を進める個人用AIアシスタントです。

音声入出力は将来拡張とし、v1では通常の文字入力を中心にします。

クラウドLLM APIの従量課金や利用回数制限には依存せず、LLM推論はローカルPC上で実行します。Web検索など外部サービスを利用する機能には、そのサービス側の制約があります。

## Architecture

    User
      ↓
    J.A.R.V.I.S. Terminal UI
      ↓
    Session Manager
      ↓
    Agent Runtime
      ↓
    Request Classifier (conversation / task only)
      ↓
    LM Studio OpenAI-compatible API
      ↓
    Gemma 4 E4B QAT
      ↓
    Tool Calling
      ↓
    Tool Registry / Dispatcher
      ├─ Filesystem
      ├─ PowerShell
      ├─ Web Search / Web Page
      ├─ Python
      └─ Memory
      ↓
    Safety Policy / Tool Execution
      ↓
    Observation / Loop Guard / Recovery
      ↓
    Completion Verification
      ↓
    Agent decides next action
      ↓
    Retry / Continue
      ↓
    Final answer

LM Studio provides the local model server and OpenAI-compatible API. The Agent Runtime remains responsible for task state, safety, tool execution, and autonomous iteration. LM Studio's OpenAI-compatible API is intentionally used here because it lets the project use LM Studio as the inference layer without transferring safety and workspace policy into the model server.

LM Studio also provides a native Python SDK and an `.act()` automatic multi-round agent API. The current JARVIS core keeps its own Runtime loop instead of delegating execution to `.act()`, because the project needs centralized workspace boundaries, safety/approval policy, observation and loop control, recovery, and deterministic task verification. This avoids maintaining two competing execution-control layers.

## Current implementation

Agent Coreには、1つの依頼を独立して追跡するTaskState、Session Manager、Loop Guard、Recovery、Safety Policy、Completion Verificationを実装しています。Cross-taskの要点はSession Managerが保持し、長期Memoryは別のCapabilityとして扱います。

ToolはToolRegistryに登録され、Runtimeが実際の操作を実行します。Qwen3:8BはToolを直接実行せず、Tool呼び出しを要求し、Runtimeが安全性を確認したうえで実行結果をLLMへ返します。Session ManagerはTask履歴の圧縮、短い会話履歴、Task間の要点保持を1つの責務にまとめています。

現在の主要Tool:

- list_directory - workspace内の一覧取得
- read_file - ローカルファイル読み取り
- search_files - ローカルファイル検索
- file_mutation - ローカルファイル作成・編集・削除
- execute_command - PowerShellコマンド実行
- search_web - Web検索
- fetch_web_page - 公開Webページ本文取得
- run_python_script - 一時Python Script実行
- save_memory / search_memory - ローカルMemory
- Recipe / Plugin capability management (opt-in Experimental)

## LM Studio

LM StudioのDeveloper tabでServerを起動し、Qwen3:8Bをロードして使用します。

推奨モデル:

    qwen/qwen3-8b

APIの既定値:

    LM_STUDIO_BASE_URL=http://localhost:1234/v1
    LM_STUDIO_MODEL=qwen/qwen3-8b

モデルIDが環境によって異なる場合は、環境変数で変更できます。

PowerShell:

    $env:LM_STUDIO_MODEL="実際のモデルID"

### Why the Agent uses the OpenAI-compatible API

LM Studio officially supports the OpenAI-compatible `/v1/chat/completions` endpoint, including custom function tools. Existing OpenAI client code can point its `base_url` at LM Studio. This keeps the integration small and lets LM Studio handle local inference and Tool Call parsing, while J.A.R.V.I.S. keeps responsibility for execution policy and safety.

The native `lmstudio-python` SDK remains a useful future option for features that specifically benefit from LM Studio's SDK, such as model lifecycle management or the SDK's built-in `.act()` agent flow. It is not duplicated in v1 while the custom Runtime remains the source of truth for execution.

## Requirements

- Windows
- Python 3.12+
- LM Studio
- Gemma 4 E4B QAT（既定）
- OpenAI Python SDK
- ddgs

Ollamaは不要です。

## Setup

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    pip install -r requirements.txt

## Run

Agentを操作したい作業ディレクトリで起動します。

    python agent.py

起動時にはModel、workspace、Auto承認ルール数が表示されます。

基本操作:

    You > WpfGisLearningを確認して、テストを実行して問題があれば修正して。

JARVIS v1では、ユーザーの1回の依頼に対して必要なToolを複数回使います。Taskとして分類された依頼では登録済みToolをモデルに提示し、LM Studio上のローカルLLMのTool Callingに選択を委ね、Runtimeが安全性・実行・観測・回復・完了確認を担当します。

## Commands

- /tasks - Task一覧
- /permissions - 学習済み承認ルール
- /clear-permissions - 承認ルール削除
- /clear-context - Session Context削除
- /exit - 終了

## Safety

- workspace内の相対パスはworkspace外へ脱出できないよう制限
- 明示されたローカル絶対パスはFile Toolで扱える
- 変更・実行系Toolには確認ポリシーを設定可能
- Auto Modeでは安全と判定された通常操作を確認なしで実行
- 高リスク操作は確認または拒否
- ユーザーが明示的に許可した操作パターンはローカル承認Policyへ保存
- execute_commandにはtimeoutを設定
- run_python_scriptは子プロセスで実行し、時間・サイズを制限
- Pluginは検疫・検証を経て有効化
- Tool結果のサイズを制限してLLMへ返す

Safety判定は完全なセキュリティサンドボックスではありません。信頼できないMCPやPluginは追加しないでください。

## Development

    python -m pytest -q

GitHub ActionsではWindows Runner上でテストします。

## Roadmap

### JARVIS Core
- Agent loopの収束・安定化
- Deterministic Completion Verification
- AGENTS.md階層ルール
- Request Classifierの維持・軽量化
- Task/Observation状態の簡素化
- より高度なLong-term Memory
- Goal / task decomposition
- Proactive behavior

### Capabilities
- Git
- Web / HTTP
- Database
- Windows automation
- Scheduler
- Notifications
- MCP integration

### Interaction
- GUI
- 常駐 / event-driven execution
- 音声入力 / 出力
- より自然なJARVIS UI

## Design principles

- LLMは判断しToolを選択する
- Runtimeは状態・安全性・実行・進捗を管理する
- Toolは明確な責務を持つ
- Agent Coreに特定用途のロジックを埋め込まない
- ToolはRegistry経由で追加する
- 同じTool + 同じ引数の重複実行を検知する
- 観測の新規性と目的へのProgressを分離する
- workspace外の意図しないアクセスを防ぐ
- 変更操作は確認可能にする
- 巨大なAgent Frameworkをそのまま導入せず、必要な機能を段階的に実装する


## Experimental capabilities

Recipe/Plugin support is intentionally outside the default JARVIS Core. A normal `python agent.py` startup does not initialize the Plugin directory or load enabled Plugins.

To explicitly use these capabilities from Python, opt in when building the Tool Registry:

    registry = create_default_tool_registry(enable_experimental=True)

The normal Runtime path does not automatically persist successful temporary Python scripts as Recipes or inject Recipe promotion candidates into every task. This keeps the Core focused on model-driven Tool selection, execution safety, observation/recovery, and deterministic completion verification.
