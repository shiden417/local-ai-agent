from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from agent.approval import ApprovalPolicy, approval_key
from agent.command_policy import AUTO_ALLOW, AUTO_DENY, classify_auto_mode
from agent.context import ContextManager
from agent.conversation import ConversationManager
from agent.llm import ask_llm
from agent.loop_guard import ToolLoopGuard
from agent.observation import truncate_text
from agent.plugin_manager import PluginManager
from agent.progress import evaluate_progress
from agent.recovery import classify_tool_outcome, recovery_guidance
from agent.session_context import SessionContext
from agent.environment import build_environment_context, extract_related_paths
from agent.recipe_store import RecipeStore
from agent.terminal_ui import TerminalUI
from agent.safety import requires_confirmation
from agent.task import TaskState
from agent.task_manager import ManagedTask, TaskManager
from agent.capability_router import Capability
from agent.tool_registry import ToolRegistry
from agent.tools import create_default_tool_registry


SYSTEM_PROMPT = """あなたはローカルで動作する汎用AI Agentです。
ユーザーの目的を達成するために、利用可能なToolを選択し、観測結果を確認しながら段階的に行動してください。

実行ルール:
- Goalを達成するために必要な最小限のActionだけを選択してください。
- 1回の判断では、原則として最も直接的なToolを1つだけ選んでください。
- PLANでは最初の具体的なActionを決め、ACTではそれを実行し、VERIFYでは結果から「目的が達成済みか」「次に何をすべきか」を判断してください。
- Current task execution stateはRuntimeが管理する事実です。Recent observationsは既知の情報として扱い、同じ情報を再取得しないでください。
- Session Contextは前のTaskから引き継いだ要点です。現在のユーザー発言と矛盾する場合は現在の発言を優先してください。
- EnvironmentはRuntimeが取得した現在の実行環境の事実です。Workspace、現在日時、Git状態、AGENTS.mdのルールを推測で置き換えないでください。
- 同じTool + 同じ引数を繰り返さないでください。Toolが無効化されている場合は別のActionを選択してください。
- 同じ内容の観測を別の引数で再取得することも避けてください。
- Toolが失敗した場合は、同じ失敗を繰り返さず、Runtimeが提示するRecovery Guideを確認して、直前の失敗に直接関係する最小の別手段を試してください。
- search_webは現在または未来の外部情報が必要なときの読み取り専用Web検索です。今日・明日・現在の天気、最新ニュース、価格、営業時間、運行状況などでは積極的に使用してください。検索結果だけで具体的な内容を確認できない場合は、結果URLをfetch_web_pageで取得して本文を確認してください。検索結果やWeb本文にない数値・事実を推測しないでください。
- run_python_scriptは一時的な補助手段です。専用Toolで目的を達成できる場合は、専用Toolを優先してください。
- Runtimeが提示した過去のRecipeは成功実績のある参考コードですが、パス・入力・出力は現在のTaskに合わせて見直してください。
- run_python_scriptが成功した場合、そのScriptはRuntimeがRecipeとして自動保存します。これを理由にsave_memoryを追加で呼ばないでください。
- 新しい永続Capabilityが必要な場合は、まずlist_promotion_candidatesで候補を確認できます。候補をそのまま自動有効化せず、必要性が明確ならgenerate_pluginで候補を生成し、test_plugin_candidateで確認付き検証を行い、成功後にstage_pluginで検疫へ配置し、最後にpromote_pluginで有効化してください。
- PluginはAgent Coreの代替ではありません。Runtime、Safety、Task、MemoryなどCore自体をPluginで変更しようとしないでください。
- Pluginを有効化した後は、次の反復で新しいToolがRegistryへ追加されるため、必要ならそのToolを使って元のGoalを続行してください。
- 現在日時・時刻についてはRuntimeが提供する現在の日時を事実として使用し、推測や古い知識から日付を作らないでください。
- ユーザーが「作成して」「修正して」「削除して」「実行して」など、実際の操作を明示した場合は、説明やサンプルだけを返さず、適切なToolを使ってください。
- Toolを使っていない場合、ファイル作成・変更・コマンド実行などが完了したと主張しないでください。
- Direct/Openの通常会話では、Toolを使おうとせず、ユーザーの発言に自然な文章で回答してください。
- コマンド失敗の調査で、実行ポリシー、System32、Windows内部ファイル、ユーザーディレクトリなどの無関係なOS情報を探索しないでください。必要性がユーザーの依頼から明確でない限り、workspace内の原因調査を優先してください。
- Pythonプロジェクトのテストでは、まず現在のプロジェクト環境を使う「python -m pytest」形式を優先してください。
- 変更や外部作用を伴うToolは、必要性を確認してから使用してください。
- finish_taskは、Goalが達成済み、または安全に進められないことが明確になったときだけ使用してください。
- ask_userは、推測で進めると誤る重要な選択肢が残っている場合だけ使用してください。質問は1つに絞ってください。
- ファイル内に記載された相対パスは、そのファイルが存在するディレクトリを基準に解決してください。workspace rootを勝手に基準にしてパスを推測しないでください。
- ユーザーが求めていない変更を行わないでください。
- 目的を達成するための十分な情報が揃ったら、Toolを追加実行せず通常の文章で直接回答してください。
- 現在または未来の情報が必要なのに、その情報を取得するToolが利用可能でない場合は、推測せず、その制約を明示してください。
- search_webの結果だけでは詳細が不足している場合、finish_taskへ進まずfetch_web_pageまたは別検索で必要な根拠を集めてください。
- 空のJSON、空配列、Tool結果そのもののコピーを最終回答にしないでください。

workspace調査のルール:
- 現在のworkspaceを調べる依頼では、まずlist_directoryで構造を確認します。
- 既知のファイルを説明する必要がある場合はread_fileを使います。
- search_filesは具体的な文字列・シンボル・識別子を探す場合だけ使用します。「主要なファイル」のような自然言語カテゴリを検索語にしないでください。
- search_memoryは過去の保存情報が今回のGoalに必要な場合だけ使用します。現在のworkspaceの調査には使用しません。

重要:
- LLMは判断し、Runtimeが状態・安全性・進捗を管理し、Toolが実際の操作を行います。
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


def _observation_fingerprint(
    tool_name: str,
    result: dict[str, Any],
) -> str:
    """Return a semantic-ish fingerprint for meaningful observation identity."""
    normalized = dict(result)

    # Different read ranges can produce the same useful information. Do not
    # treat range metadata alone as a new observation.
    if tool_name == "read_file":
        normalized.pop("start_line", None)
        normalized.pop("end_line", None)

    # Ignore common metadata that does not represent task knowledge.
    for key in ("timestamp", "created_at", "updated_at", "duration_ms"):
        normalized.pop(key, None)

    payload = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class AgentRuntime:
    def __init__(
        self,
        working_directory: str | Path,
        max_iterations: int = 10,
        tool_registry: ToolRegistry | None = None,
        confirm: Callable[[str], bool] | None = None,
        context_manager: ContextManager | None = None,
        recipe_store: RecipeStore | None = None,
        plugin_manager: PluginManager | None = None,
        approval_policy: ApprovalPolicy | None = None,
        terminal_ui: TerminalUI | None = None,
        ask_user: Callable[[str], str] | None = None,
    ) -> None:
        self.working_directory = Path(working_directory).resolve()
        self.max_iterations = max_iterations
        self.plugin_manager = plugin_manager or PluginManager()
        self.recipe_store = recipe_store or RecipeStore()
        self.tool_registry = tool_registry or create_default_tool_registry(
            plugin_manager=self.plugin_manager,
            recipe_store=self.recipe_store,
        )
        self.confirm = confirm
        self.approval_policy = approval_policy or ApprovalPolicy()
        self.terminal_ui = terminal_ui
        self.ask_user_callback = ask_user
        self.context_manager = context_manager or ContextManager()
        self.loop_guard = ToolLoopGuard()
        self.task_manager = TaskManager()
        self.conversation_manager = ConversationManager()
        self.session_context = SessionContext()
        self.current_task: ManagedTask | None = None
        self.task: TaskState | None = None

    @property
    def messages(self) -> list[dict[str, Any]]:
        """Return the current task history for compatibility and inspection."""
        if self.current_task is None:
            return [{"role": "system", "content": SYSTEM_PROMPT}]
        return self.current_task.messages

    @staticmethod
    def _default_confirm(message: str) -> bool:
        answer = input(f"\n{message}\nProceed? [y/N]: ")
        return answer.strip().lower() in {"y", "yes"}

    def _request_confirmation(
        self,
        summary: str,
        permission_key: str,
    ) -> bool:
        if self.confirm is not None:
            return bool(self.confirm(summary))

        if self.terminal_ui is not None:
            answer = self.terminal_ui.approval(summary)
        else:
            print(f"\n{summary}")
            answer = input(
                "Approval? [y] once / [a] always for this action / [n] deny: "
            ).strip().lower()
        if answer in {"a", "always"}:
            self.approval_policy.allow(permission_key, summary)
            print("[Approval] learned")
            return True
        return answer in {"y", "yes"}
    def run(self, user_input: str) -> str:
        route = self.tool_registry.route_for(user_input)
        if route.mode.value in {"direct", "open"}:
            return self._run_conversation(user_input)

        current_task = self.task_manager.create(user_input)
        current_task.messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input},
        ]

        self.current_task = current_task
        self.task = current_task.state
        self.loop_guard.reset()
        terminal_synthesis_required = False
        self.task.start()
        self.task_manager.update_timestamp(current_task)
        if self.terminal_ui is not None:
            self.terminal_ui.task_start(current_task.goal, current_task.task_id)

        for _ in range(self.max_iterations):
            self.task.begin_iteration()
            self.task_manager.update_timestamp(current_task)
            if self.terminal_ui is not None:
                self.terminal_ui.phase(self.task.phase.value, self.task.iteration)

            context_messages = self.context_manager.prepare(current_task.messages)
            prior_conversation = self.conversation_manager.recent_messages()
            related_paths = self._related_paths(current_task)
            route = self.tool_registry.route_for(self.task.goal)
            capability_text = ", ".join(
                capability.value for capability in sorted(
                    route.capabilities,
                    key=lambda item: item.value,
                )
            ) or "none"
            recipe_messages: list[dict[str, Any]] = []
            if Capability.CAPABILITY_MANAGEMENT in route.capabilities:
                promotion_candidates = self.recipe_store.promotion_candidates(
                    min_uses=2
                )
                if promotion_candidates:
                    candidate_lines = [
                        "Recipe promotion candidates (do not promote automatically):"
                    ]
                    for candidate in promotion_candidates[:3]:
                        candidate_lines.append(
                            f"- id={candidate.id}, use_count={candidate.use_count}, "
                            f"goal={candidate.goal}"
                        )
                    recipe_messages.append(
                        {
                            "role": "system",
                            "content": "\n".join(candidate_lines),
                        }
                    )
            if Capability.SCRIPT_EXECUTION in route.capabilities:
                recipes = self.recipe_store.search(self.task.goal, limit=2)
                if recipes:
                    recipe_lines = [
                        "Relevant successful local Recipes (reference only):"
                    ]
                    for index, recipe in enumerate(recipes, start=1):
                        bounded_script, _ = truncate_text(recipe.script, 4_000)
                        recipe_lines.append(
                            f"Recipe {index}: use_count={recipe.use_count}, "
                            f"goal={recipe.goal}"
                        )
                        recipe_lines.append(bounded_script)
                    recipe_messages.append(
                        {
                            "role": "system",
                            "content": "\n".join(recipe_lines),
                        }
                    )

            task_system_messages = [
                message
                for message in context_messages
                if message.get("role") == "system"
            ]
            task_non_system_messages = [
                message
                for message in context_messages
                if message.get("role") != "system"
            ]

            current_datetime = datetime.now().astimezone().isoformat(timespec="seconds")
            environment_context = build_environment_context(
                self.working_directory,
                related_paths,
            )

            llm_messages = [
                *task_system_messages,
                {"role": "system", "content": environment_context},
                {"role": "system", "content": self.session_context.prompt_block()},
                *recipe_messages,
                *prior_conversation,
                *task_non_system_messages,
                {
                    "role": "system",
                    "content": (
                        "Current local date/time (Runtime authoritative): "
                        f"{current_datetime}\n"
                        "Current task execution dashboard. "
                        "Treat this as Runtime-managed state; do not reconstruct "
                        "progress only from chat history.\n"
                        f"Tool scope={route.mode.value}; "
                        f"capabilities={capability_text}\n"
                        f"{self.task.snapshot()}\n"
                        "Tool use is optional. In scoped/open modes, call a tool "
                        "only when it advances the goal; otherwise answer directly."
                    ),
                },
            ]

            excluded_tools = set(self.task.disabled_tools)
            if self.task.recovery_tool:
                excluded_tools.add(self.task.recovery_tool)

            available_tools = self.tool_registry.schemas_for(
                self.task.goal,
                excluded_tools=excluded_tools,
            )

            force_synthesis = (
                terminal_synthesis_required
                or self.task.no_progress_streak >= 2
                or not available_tools
            )
            if self.task.recovery_tool:
                llm_messages.append(
                    {
                        "role": "system",
                        "content": (
                            f"Recovery mode: the previous Tool "
                            f"'{self.task.recovery_tool}' failed. "
                            "Do not use that Tool in the next step. "
                            "Use a directly relevant alternative observation "
                            "within the workspace, then reassess the goal. "
                            "Do not investigate unrelated OS settings or "
                            "system internals."
                        ),
                    }
                )
            if self.task.last_failure_status:
                llm_messages.append(
                    {
                        "role": "system",
                        "content": recovery_guidance(
                            self.task.last_tool or "unknown",
                            self.task.last_failure_status,
                        ),
                    }
                )

            if force_synthesis:
                llm_messages.append(
                    {
                        "role": "system",
                        "content": (
                            "Do not call any more tools in this turn. "
                            "Synthesize the best direct answer from the "
                            "observations already available and answer the "
                            "user directly."
                        ),
                    }
                )
                available_tools = []

            if self.terminal_ui is not None:
                self.terminal_ui.thinking_start()

            try:
                response = ask_llm(
                    llm_messages,
                    tools=available_tools,
                )
            finally:
                if self.terminal_ui is not None:
                    self.terminal_ui.thinking_stop()
            message = response.choices[0].message
            tool_calls = getattr(message, "tool_calls", None) or []
            content = self._normalize_final_content(
                getattr(message, "content", None) or ""
            )

            if not tool_calls:
                if (
                    route.mode.value == "scoped"
                    and self.task.tool_calls == 0
                    and self.task.iteration == 1
                ):
                    current_task.messages.append(_message_to_dict(message))
                    current_task.messages.append(
                        {
                            "role": "system",
                            "content": (
                                "The user gave an operational request. "
                                "Do not answer with instructions, examples, or a "
                                "claim that the work is complete without executing "
                                "the appropriate Tool. Use a Tool now. If the "
                                "request lacks one required detail, ask only a "
                                "concise clarification question."
                            ),
                        }
                    )
                    continue

                if self._is_invalid_final_response(
                    content,
                    current_task.messages,
                ):
                    current_task.messages.append(_message_to_dict(message))
                    current_task.messages.append(
                        {
                            "role": "system",
                            "content": (
                                "The previous response was empty or only an empty "
                                "JSON container. Continue using the Runtime "
                                "dashboard and provide a direct answer."
                            ),
                        }
                    )
                    continue

                current_task.messages.append(_message_to_dict(message))
                final_content = self._normalize_final_content(content)
                self.conversation_manager.add_turn(
                    user_input,
                    final_content,
                )
                self.session_context.remember_task(
                    user_input,
                    final_content,
                    current_task.messages,
                )
                self.task.complete()
                self.task_manager.update_timestamp(current_task)
                if self.terminal_ui is not None:
                    self.terminal_ui.final(final_content)
                return final_content

            current_task.messages.append(_message_to_dict(message))

            for tool_call in tool_calls:
                if terminal_synthesis_required:
                    break

                try:
                    call_id, name, arguments = _tool_call_values(tool_call)
                except (TypeError, ValueError, json.JSONDecodeError) as exc:
                    print(f"[Tool Error] {exc}")
                    self.task.record_tool(
                        "invalid_tool_call",
                        succeeded=False,
                        summary=str(exc),
                        signature=f"invalid_tool_call:{type(exc).__name__}:{exc}",
                        new_information=False,
                        progress_state=evaluate_progress(
                            "invalid_tool_call",
                            {"ok": False, "error": str(exc)},
                            observation_is_new=False,
                        ).state,
                    )
                    current_task.messages.append(
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

                call_count = self.loop_guard.record(name, arguments)
                if self.task.recovery_tool == name:
                    result = {
                        "ok": False,
                        "error": (
                            f"Tool '{name}' is temporarily blocked during "
                            "failure recovery. Use a directly relevant "
                            "alternative Tool first."
                        ),
                        "recovery_blocked": True,
                    }
                    if self.terminal_ui is not None:
                        self.terminal_ui.info(f"Tool blocked by recovery quarantine: {name}")
                    else:
                        print("[Tool] blocked by recovery quarantine")
                elif self.loop_guard.is_repetition(name, arguments):
                    self.task.disable_tool(name)
                    result = {
                        "ok": False,
                        "error": self.loop_guard.message(name, arguments),
                        "repeated_tool_call": True,
                        "call_count": call_count,
                    }
                    if self.terminal_ui is not None:
                        self.terminal_ui.info(f"Repeated Tool blocked: {name}")
                    else:
                        print("[Tool] repeated call blocked; tool disabled for this task")
                elif requires_confirmation(
                    name,
                    arguments,
                    self.tool_registry,
                    self.working_directory,
                ):
                    auto_decision = classify_auto_mode(
                        name, arguments, self.tool_registry, self.working_directory
                    )
                    if auto_decision == AUTO_DENY:
                        result = {
                            "ok": False,
                            "error": "Auto Mode blocked this high-risk operation.",
                            "auto_mode": "deny",
                        }
                        if self.terminal_ui is not None:
                            self.terminal_ui.info(f"Auto Mode blocked: {name}")
                        else:
                            print(f"[Auto Mode] blocked: {name}")
                    elif self.confirm is None and auto_decision == AUTO_ALLOW:
                        result = self._execute_tool(name, arguments)
                        if self.terminal_ui is not None:
                            self.terminal_ui.info(f"Auto Mode: {name}")
                    else:
                        permission_key = approval_key(
                            name, arguments, self.working_directory
                        )
                        if self.approval_policy.is_allowed(permission_key):
                            if self.terminal_ui is not None:
                                self.terminal_ui.info(f"Learned permission: {name}")
                            else:
                                print("[Approval] learned permission")
                            result = self._execute_tool(name, arguments)
                        else:
                            summary = self._confirmation_message(name, arguments)
                            decision = self._request_confirmation(
                                summary, permission_key
                            )
                            if not decision:
                                result = {
                                    "ok": False,
                                    "error": "User rejected the operation.",
                                    "user_rejected": True,
                                }
                                if self.terminal_ui is not None:
                                    self.terminal_ui.info(f"Approval rejected: {name}")
                                else:
                                    print("[Tool] rejected by user")
                            else:
                                result = self._execute_tool(name, arguments)
                else:
                    result = self._execute_tool(name, arguments)

                if name == "ask_user" and bool(result.get("ok")):
                    question = str(result.get("question", "")).strip()
                    answer = self._ask_user(question)
                    if answer is None:
                        result = {
                            "ok": False,
                            "error": "User did not provide an answer.",
                            "user_rejected": True,
                        }
                    else:
                        result = {**result, "answer": answer}

                if name == "finish_task" and bool(result.get("ok")):
                    verification_error = self._verify_finish_task(
                        current_task,
                        result,
                    )
                    if verification_error is not None:
                        result = {
                            **result,
                            "ok": False,
                            "verification_failed": True,
                            "error": verification_error,
                        }
                    else:
                        terminal_synthesis_required = True

                result["status"] = classify_tool_outcome(name, result)

                serialized = json.dumps(
                    result,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                bounded, truncated = truncate_text(serialized)

                observation_summary = self._observation_summary(result)
                fingerprint = _observation_fingerprint(name, result)
                signature = f"{name}:{fingerprint}"
                observation_is_new = signature not in self.task.observation_signatures
                outcome_status = classify_tool_outcome(name, result)
                evaluation = evaluate_progress(
                    name,
                    result,
                    observation_is_new=observation_is_new,
                )

                self.task.record_tool(
                    name,
                    succeeded=bool(result.get("ok")),
                    summary=observation_summary,
                    signature=signature,
                    new_information=observation_is_new,
                    progress_state=evaluation.state,
                    failure_status=outcome_status if not bool(result.get("ok")) else None,
                )
                self.task_manager.update_timestamp(current_task)

                if self.terminal_ui is not None:
                    self.terminal_ui.tool_result(
                        bool(result.get("ok")),
                        observation_summary,
                    )
                    if truncated:
                        self.terminal_ui.info("Tool result was truncated before returning to the model.")
                else:
                    print("[Result]")
                    print(bounded)
                    if truncated:
                        print("[Result] output truncated before returning to the model.")

                current_task.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": name,
                        "content": bounded,
                    }
                )

                if name == "finish_task" and bool(result.get("ok")):
                    if str(result.get("completion_status", "completed")) == "blocked":
                        self.task.fail(
                            str(result.get("summary", "Task could not be completed."))
                        )
                    else:
                        self.task.complete()
                    terminal_synthesis_required = True
                    break

                if name == "run_python_script" and bool(result.get("ok")):
                    script = str(arguments.get("script", "")).strip()
                    if script:
                        try:
                            recipe = self.recipe_store.record(
                                self.task.goal,
                                script,
                            )
                            print(
                                f"[Recipe] saved {recipe.id} "
                                f"(use_count={recipe.use_count})"
                            )
                        except ValueError as exc:
                            print(f"[Recipe] skipped: {exc}")

                tool_definition = self.tool_registry.get(name)
                if (
                    bool(result.get("ok"))
                    and tool_definition is not None
                    and tool_definition.terminal_on_success
                    and not any(
                        capability.value in {
                            "process",
                            "memory_read",
                            "memory_write",
                        }
                        for capability in route.capabilities
                    )
                ):
                    terminal_synthesis_required = True

        self.task.hit_max_iterations()
        self.task_manager.update_timestamp(current_task)
        return "Agentの最大反復回数に達したため、処理を終了しました。"

    def _run_conversation(self, user_input: str) -> str:
        """Answer without creating a Task or exposing operational Tools."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": build_environment_context(self.working_directory)},
            {"role": "system", "content": self.session_context.prompt_block()},
            *self.conversation_manager.recent_messages(),
            {"role": "user", "content": user_input.strip()},
        ]
        if self.terminal_ui is not None:
            self.terminal_ui.thinking_start()
        try:
            response = ask_llm(messages, tools=[])
        finally:
            if self.terminal_ui is not None:
                self.terminal_ui.thinking_stop()

        message = response.choices[0].message
        content = self._normalize_final_content(
            getattr(message, "content", None) or ""
        )
        if not content:
            content = "すみません。うまく回答を生成できませんでした。"

        self.conversation_manager.add_turn(user_input, content)
        if self.terminal_ui is not None:
            self.terminal_ui.final(content)
        return content

    def _verify_finish_task(
        self,
        task: ManagedTask,
        result: dict[str, Any],
    ) -> str | None:
        """Verify deterministic execution facts before accepting completion."""
        if str(result.get("completion_status", "")).strip().lower() == "blocked":
            return None

        successful_tools: list[tuple[str, dict[str, Any]]] = []
        for message in task.messages:
            if message.get("role") != "tool":
                continue
            try:
                payload = json.loads(str(message.get("content", "")))
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict) and payload.get("ok"):
                name = str(message.get("name", ""))
                if name and name != "finish_task":
                    successful_tools.append((name, payload))

        if not successful_tools:
            return (
                "System Verification Failed: no successful action has been observed "
                "before finish_task. Perform the required action first."
            )

        name, payload = successful_tools[-1]

        if name == "execute_command":
            if payload.get("exit_code") != 0:
                return (
                    "System Verification Failed: the last command did not finish "
                    "with exit_code 0."
                )
            return None

        if name in {"file_mutation", "create_file", "edit_file", "delete_file"}:
            path = str(payload.get("path", "")).strip()
            if not path:
                return (
                    "System Verification Failed: the file operation did not "
                    "return a target path."
                )
            target = Path(path)
            if not target.is_absolute():
                target = self.working_directory / target

            if payload.get("deleted") is True:
                if target.exists():
                    return (
                        f"System Verification Failed: target file still exists: {path}"
                    )
                return None
            if not target.exists():
                return (
                    f"System Verification Failed: target file does not exist: {path}"
                )
            return None

        if name == "fetch_web_page":
            status_code = payload.get("status_code")
            content = str(payload.get("content", "")).strip()
            if not content or (
                isinstance(status_code, int) and not 200 <= status_code < 400
            ):
                return (
                    "System Verification Failed: the fetched page did not return "
                    "usable content."
                )
            return None

        if name == "search_web":
            if int(payload.get("count", 0) or 0) <= 0:
                return (
                    "System Verification Failed: the web search returned no results."
                )
            return None

        return None

    def _related_paths(self, task: ManagedTask) -> list[str]:
        candidates = extract_related_paths(task.goal)
        for message in reversed(task.messages):
            if message.get("role") != "tool":
                continue
            try:
                result = json.loads(str(message.get("content", "")))
            except json.JSONDecodeError:
                continue
            if not isinstance(result, dict):
                continue
            for key in ("path", "directory", "relative_reference_base"):
                value = str(result.get(key, "")).strip()
                if value:
                    candidates.append(value)
        deduped: list[str] = []
        for value in candidates:
            if value not in deduped:
                deduped.append(value)
        return deduped[:12]

    @staticmethod
    def _normalize_final_content(content: Any) -> str:
        """Convert model content/message-like payloads into plain user text."""
        if content is None:
            return ""

        if isinstance(content, str):
            normalized = content.strip()
            if normalized.startswith("{") and normalized.endswith("}"):
                try:
                    decoded = json.loads(normalized)
                except json.JSONDecodeError:
                    return normalized
                if isinstance(decoded, dict) and "content" in decoded:
                    return AgentRuntime._normalize_final_content(
                        decoded["content"]
                    )
            return normalized

        if isinstance(content, dict):
            nested = content.get("content")
            if nested is not None:
                return AgentRuntime._normalize_final_content(nested)

        nested = getattr(content, "content", None)
        if nested is not None and nested is not content:
            return AgentRuntime._normalize_final_content(nested)

        return str(content).strip()

    @staticmethod
    def _is_invalid_final_response(
        content: str,
        messages: list[dict[str, Any]] | None = None,
    ) -> bool:
        normalized = content.strip()
        if not normalized or normalized in {"{}", "[]"}:
            return True

        if messages:
            return any(
                message.get("role") == "tool"
                and str(message.get("content", "")).strip() == normalized
                for message in messages
            )

        return False

    @staticmethod
    def _observation_summary(result: dict[str, Any]) -> str:
        summary = json.dumps(
            result,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        summary, _ = truncate_text(summary, 900)
        return summary

    def _ask_user(self, question: str) -> str | None:
        question = question.strip()
        if not question:
            return None
        if self.ask_user_callback is not None:
            return str(self.ask_user_callback(question)).strip()
        if self.terminal_ui is not None:
            return self.terminal_ui.question(question)
        print(f"\n[Agent Question] {question}")
        try:
            return input("Answer: ").strip()
        except EOFError:
            return None

    def list_tasks(self) -> list[ManagedTask]:
        return self.task_manager.list_tasks()

    def _execute_tool(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        if self.terminal_ui is not None:
            self.terminal_ui.tool_start(name, arguments)
        else:
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
                "Agentがローカルファイルを変更しようとしています。\n"
                f"path: {arguments.get('path', '')}"
            )

        if name == "file_mutation":
            operation = str(arguments.get("operation", "")).strip() or "変更"
            return (
                f"Agentがローカルファイルを{operation}しようとしています。\n"
                f"path: {arguments.get('path', '')}"
            )

        if name == "run_python_script":
            return (
                "Agentが一時的なPythonスクリプトを実行しようとしています。\n"
                "実行環境は子プロセスで時間・出力サイズを制限します。\n"
                f"timeout_seconds: {arguments.get('timeout_seconds', 15)}"
            )

        if name == "test_plugin_candidate":
            return (
                "Agentが生成したPlugin候補を子プロセスで実行して検証しようとしています。\n"
                f"plugin_id: {arguments.get('plugin_id', '')}"
            )

        if name == "stage_plugin":
            return (
                "Agentが新しいPluginを検疫領域へ作成しようとしています。\n"
                f"plugin_id: {arguments.get('plugin_id', '')}"
            )

        if name == "promote_plugin":
            return (
                "Agentが検疫済みPluginを永続Capabilityとして有効化しようとしています。\n"
                f"plugin_id: {arguments.get('plugin_id', '')}"
            )

        return (
            "Agentが確認の必要な操作を実行しようとしています。\n"
            f"tool: {name}\n"
            f"arguments: {json.dumps(arguments, ensure_ascii=False)}"
        )
