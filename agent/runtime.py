from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from agent.context import ContextManager
from agent.conversation import ConversationManager
from agent.llm import ask_llm
from agent.loop_guard import ToolLoopGuard
from agent.observation import truncate_text
from agent.progress import evaluate_progress
from agent.safety import requires_confirmation
from agent.task import TaskState
from agent.task_manager import ManagedTask, TaskManager
from agent.tool_registry import ToolRegistry
from agent.tools import create_default_tool_registry


SYSTEM_PROMPT = """あなたはローカルで動作する汎用AI Agentです。
ユーザーの目的を達成するために、利用可能なToolを選択し、観測結果を確認しながら段階的に行動してください。

実行ルール:
- Goalを達成するために必要な最小限のActionだけを選択してください。
- PLANでは最初の具体的なActionを決め、ACTではそれを実行し、VERIFYでは結果から「目的が達成済みか」「次に何をすべきか」を判断してください。
- Current task execution stateはRuntimeが管理する事実です。Recent observationsは既知の情報として扱い、同じ情報を再取得しないでください。
- 同じTool + 同じ引数を繰り返さないでください。Toolが無効化されている場合は別のActionを選択してください。
- 同じ内容の観測を別の引数で再取得することも避けてください。
- Toolが失敗した場合は、同じ失敗を繰り返さず、直前の失敗に直接関係する最小の別手段を試してください。
- 現在日時・時刻についてはRuntimeが提供する現在の日時を事実として使用し、推測や古い知識から日付を作らないでください。
- ユーザーが「作成して」「修正して」「削除して」「実行して」など、実際の操作を明示した場合は、説明やサンプルだけを返さず、適切なToolを使ってください。
- Toolを使っていない場合、ファイル作成・変更・コマンド実行などが完了したと主張しないでください。
- コマンド失敗の調査で、実行ポリシー、System32、Windows内部ファイル、ユーザーディレクトリなどの無関係なOS情報を探索しないでください。必要性がユーザーの依頼から明確でない限り、workspace内の原因調査を優先してください。
- Pythonプロジェクトのテストでは、まず現在のプロジェクト環境を使う「python -m pytest」形式を優先してください。
- 変更や外部作用を伴うToolは、必要性を確認してから使用してください。
- ユーザーが求めていない変更を行わないでください。
- 目的を達成するための十分な情報が揃ったら、Toolを追加実行せず通常の文章で直接回答してください。
- 現在情報が必要なのに、その情報を取得するToolが利用可能でない場合は、推測せず、その制約を明示してください。
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
    ) -> None:
        self.working_directory = Path(working_directory).resolve()
        self.max_iterations = max_iterations
        self.tool_registry = tool_registry or create_default_tool_registry()
        self.confirm = confirm or self._default_confirm
        self.context_manager = context_manager or ContextManager()
        self.loop_guard = ToolLoopGuard()
        self.task_manager = TaskManager()
        self.conversation_manager = ConversationManager()
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

    def run(self, user_input: str) -> str:
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

        for _ in range(self.max_iterations):
            self.task.begin_iteration()
            self.task_manager.update_timestamp(current_task)

            context_messages = self.context_manager.prepare(current_task.messages)
            prior_conversation = self.conversation_manager.recent_messages()
            route = self.tool_registry.route_for(self.task.goal)
            capability_text = ", ".join(
                capability.value for capability in sorted(
                    route.capabilities,
                    key=lambda item: item.value,
                )
            ) or "none"

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

            llm_messages = [
                *task_system_messages,
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

            response = ask_llm(
                llm_messages,
                tools=available_tools,
            )
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
                self.task.complete()
                self.task_manager.update_timestamp(current_task)
                return final_content

            current_task.messages.append(_message_to_dict(message))

            for tool_call in tool_calls:
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
                    print("[Tool] blocked by recovery quarantine")
                elif self.loop_guard.is_repetition(name, arguments):
                    self.task.disable_tool(name)
                    result = {
                        "ok": False,
                        "error": self.loop_guard.message(name, arguments),
                        "repeated_tool_call": True,
                        "call_count": call_count,
                    }
                    print("[Tool] repeated call blocked; tool disabled for this task")
                elif requires_confirmation(
                    name,
                    arguments,
                    self.tool_registry,
                    self.working_directory,
                ):
                    summary = self._confirmation_message(name, arguments)
                    if not self.confirm(summary):
                        result = {
                            "ok": False,
                            "error": "User rejected the operation.",
                            "user_rejected": True,
                        }
                        print("[Tool] rejected by user")
                    else:
                        result = self._execute_tool(name, arguments)
                else:
                    result = self._execute_tool(name, arguments)

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
                )
                self.task_manager.update_timestamp(current_task)

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

    def list_tasks(self) -> list[ManagedTask]:
        return self.task_manager.list_tasks()

    def _execute_tool(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
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

        return (
            "Agentが確認の必要な操作を実行しようとしています。\n"
            f"tool: {name}\n"
            f"arguments: {json.dumps(arguments, ensure_ascii=False)}"
        )
