from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from agent.task_requirements import classify_task_requirements


class CompletionVerifier:
    """Deterministically verify that an Agent task has concrete evidence of completion."""

    def __init__(self, working_directory: str | Path) -> None:
        self.working_directory = Path(working_directory).resolve()

    def verify(
        self,
        task: Any,
        result: dict[str, Any],
        *,
        goal_text: str | None = None,
    ) -> str | None:
        completion_status = str(result.get("completion_status", "")).strip().lower()
        verification_goal = goal_text or task.goal
        messages = getattr(task, "messages", None) or []

        if completion_status == "blocked":
            if self._requires_process_execution(verification_goal) and not self._has_successful_command(messages):
                return (
                    "System Verification Failed: this task explicitly requires a process/command execution, "
                    "but no successful execute_command result was observed. Do not mark the task blocked; "
                    "use the available execution Tool first."
                )
            return None

        mutation_required = self._requires_file_mutation(verification_goal)
        process_required = self._requires_process_execution(verification_goal)

        if mutation_required:
            mutation_tools = {"file_mutation", "create_file", "edit_file", "delete_file"}
            successful_mutation = any(
                message.get("role") == "tool"
                and message.get("name") in mutation_tools
                and self._tool_payload_ok(message)
                for message in messages
            )
            if not successful_mutation:
                return (
                    "System Verification Failed: no successful action has been observed "
                    "before finish_task; this task explicitly requires a file change. "
                    "No successful file mutation was observed. "
                    "Perform the requested file mutation first."
                )

            if self._requires_test_verification(verification_goal) and not self._has_successful_test(messages):
                return (
                    "System Verification Failed: this task explicitly requires testing/verification, "
                    "but no successful pytest/test execution was observed after the file change. "
                    "Run the relevant test command and inspect its result before completion."
                )

        if process_required and not self._has_successful_command(messages):
            return (
                "System Verification Failed: this task explicitly requires process/command execution, "
                "but no successful execute_command result with exit_code 0 was observed. "
                "Use execute_command and verify its result before completion."
            )

        if self._is_read_only_request(verification_goal):
            mutation_tools = {"file_mutation", "create_file", "edit_file", "delete_file"}
            if any(
                message.get("role") == "tool"
                and message.get("name") in mutation_tools
                and self._tool_payload_ok(message)
                for message in messages
            ):
                return (
                    "System Verification Failed: this task explicitly requested "
                    "read-only investigation, but a file mutation was executed. "
                    "Do not claim the task completed successfully."
                )

        successful_tools: list[tuple[str, dict[str, Any]]] = []
        for message in messages:
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

        if self._requires_python_test_after_mutation(messages):
            return (
                "System Verification Failed: a Python file was changed in a pytest project. "
                "Run 'python -m pytest' successfully after the latest Python file change "
                "before declaring completion."
            )

        if self._requires_project_diagnostic(verification_goal):
            diagnostic_tools = {"read_file", "search_files", "execute_command"}
            if not any(
                tool_name in diagnostic_tools
                for tool_name, _ in successful_tools
            ):
                return (
                    "System Verification Failed: this project investigation needs "
                    "at least one concrete diagnostic action such as reading/searching "
                    "source files or running a relevant command/test before completion."
                )

        if not successful_tools:
            return (
                "System Verification Failed: no successful action has been observed "
                "before finish_task. Perform the required action first."
            )

        name, payload = successful_tools[-1]

        if (
            self._requires_primary_web_source(verification_goal)
            and any(tool_name == "search_web" for tool_name, _ in successful_tools)
            and not any(tool_name == "fetch_web_page" for tool_name, _ in successful_tools)
        ):
            return (
                "System Verification Failed: this request asks for current, official, "
                "release, or change-specific information. Fetch a relevant source page "
                "after search_web before declaring completion."
            )

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

    @staticmethod
    def _tool_payload_ok(message: dict[str, Any]) -> bool:
        try:
            payload = json.loads(str(message.get("content", "")))
        except json.JSONDecodeError:
            return False
        return isinstance(payload, dict) and bool(payload.get("ok"))

    @staticmethod
    def _has_successful_command(messages: list[dict[str, Any]]) -> bool:
        for message in messages:
            if message.get("role") != "tool" or message.get("name") != "execute_command":
                continue
            try:
                payload = json.loads(str(message.get("content", "")))
            except json.JSONDecodeError:
                continue
            if (
                isinstance(payload, dict)
                and payload.get("ok")
                and payload.get("exit_code") == 0
            ):
                return True
        return False

    @staticmethod
    def _has_successful_test(messages: list[dict[str, Any]]) -> bool:
        for message in messages:
            if message.get("role") != "tool":
                continue
            if message.get("name") not in {"execute_command", "run_python_script"}:
                continue
            try:
                payload = json.loads(str(message.get("content", "")))
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict) or not payload.get("ok"):
                continue
            if payload.get("exit_code") not in (None, 0):
                continue
            command = str(payload.get("command", "")).casefold()
            output = "\n".join(
                str(payload.get(key, ""))
                for key in ("stdout", "stderr")
            ).casefold()
            if "pytest" in command or "pytest" in output:
                return True
            if re.search(r"\btest(?:ing|s)?\b", command) and re.search(
                r"pass|success",
                output,
            ):
                return True
            if re.search(r"\b\d+\s+passed\b", output):
                return True
        return False

    @staticmethod
    def _requires_file_mutation(goal: str) -> bool:
        return classify_task_requirements(goal).file_mutation

    @staticmethod
    def _requires_test_verification(goal: str) -> bool:
        return classify_task_requirements(goal).test_verification

    @staticmethod
    def _requires_process_execution(goal: str) -> bool:
        return classify_task_requirements(goal).process_execution

    @staticmethod
    def _is_read_only_request(goal: str) -> bool:
        return classify_task_requirements(goal).read_only

    def _requires_python_test_after_mutation(
        self,
        messages: list[dict[str, Any]],
    ) -> bool:
        """Require a successful pytest run after the latest Python mutation."""
        if not (self.working_directory / "pytest.ini").exists() and not (
            self.working_directory / "tests"
        ).is_dir():
            return False

        latest_python_mutation_index: int | None = None
        for index, message in enumerate(messages):
            if message.get("role") != "tool":
                continue
            name = str(message.get("name", ""))
            if name not in {"file_mutation", "create_file", "edit_file"}:
                continue
            try:
                payload = json.loads(str(message.get("content", "")))
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict) or not payload.get("ok"):
                continue
            if payload.get("deleted") is True:
                continue
            path = str(payload.get("path", "")).strip()
            if path.lower().endswith(".py"):
                latest_python_mutation_index = index

        if latest_python_mutation_index is None:
            return False

        for message in messages[latest_python_mutation_index + 1 :]:
            if message.get("role") != "tool" or message.get("name") != "execute_command":
                continue
            try:
                payload = json.loads(str(message.get("content", "")))
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            command = str(payload.get("command", "")).casefold()
            if payload.get("ok") and payload.get("exit_code") == 0 and "pytest" in command:
                return False

        return True

    @staticmethod
    def _requires_project_diagnostic(goal: str) -> bool:
        text = str(goal).casefold()
        has_project_context = bool(
            re.search(r"(プロジェクト|workspace|repository|repo|コード)", text)
        )
        has_diagnostic_intent = bool(
            re.search(r"(問題点|問題|不具合|バグ|原因|調査|分析|診断)", text)
        )
        return has_project_context and has_diagnostic_intent

    @staticmethod
    def _requires_primary_web_source(goal: str) -> bool:
        text = str(goal).casefold()
        return bool(
            re.search(
                r"(最新|最新版|現在|公式|公式情報|リリース|release|変更点|主な変更|what'?s new|latest|current)",
                text,
            )
        )
