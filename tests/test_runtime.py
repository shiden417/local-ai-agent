import json
from pathlib import Path
from types import SimpleNamespace

from agent.runtime import AgentRuntime
from agent.tool_registry import ToolDefinition, ToolRegistry
import agent.runtime as runtime_module


def test_runtime_initializes_with_absolute_working_directory(tmp_path: Path) -> None:
    runtime = AgentRuntime(tmp_path)

    assert runtime.working_directory.is_absolute()
    assert runtime.max_iterations == 10
    assert runtime.messages[0]["role"] == "system"
    assert set(runtime.tool_registry.names()) == {
        "list_directory",
        "read_file",
        "search_files",
        "search_web",
        "fetch_web_page",
        "ask_user",
        "finish_task",
        "file_mutation",
        "run_python_script",
        "execute_command",
        "save_memory",
        "search_memory",
    }


def test_runtime_executes_tool_then_returns_final_response(
    tmp_path: Path,
    monkeypatch,
) -> None:
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="hello",
            description="Return a greeting",
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
            },
            handler=lambda _working_directory, _arguments: {
                "ok": True,
                "message": "hello from tool",
            },
        )
    )

    tool_call = SimpleNamespace(
        id="call-1",
        function=SimpleNamespace(
            name="hello",
            arguments="{}",
        ),
    )
    tool_message = SimpleNamespace(
        role="assistant",
        content="",
        tool_calls=[tool_call],
    )
    final_message = SimpleNamespace(
        role="assistant",
        content="作業が完了しました。",
        tool_calls=[],
    )

    responses = [
        SimpleNamespace(
            choices=[SimpleNamespace(message=tool_message)]
        ),
        SimpleNamespace(
            choices=[SimpleNamespace(message=final_message)]
        ),
    ]

    def fake_ask_llm(messages, tools):
        assert tools[0]["function"]["name"] == "hello"
        assert messages
        return responses.pop(0)

    monkeypatch.setattr(runtime_module, "ask_llm", fake_ask_llm)

    runtime = AgentRuntime(
        tmp_path,
        tool_registry=registry,
        confirm=lambda _message: True,
    )

    result = runtime.run("調査してください")

    assert result == "作業が完了しました。"
    assert runtime.task is not None
    assert runtime.task.status.value == "completed"
    assert runtime.task.tool_calls == 1
    assert any(
        message.get("role") == "tool"
        and message.get("tool_call_id") == "call-1"
        for message in runtime.messages
    )


def test_runtime_rejects_mutating_tool_before_execution(
    tmp_path: Path,
    monkeypatch,
) -> None:
    registry = ToolRegistry()
    executed = {"value": False}

    def mutate(_working_directory, _arguments):
        executed["value"] = True
        return {"ok": True}

    registry.register(
        ToolDefinition(
            name="edit_file",
            description="Mutate a file",
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
            },
            handler=mutate,
            requires_confirmation=True,
        )
    )

    tool_call = SimpleNamespace(
        id="call-edit",
        function=SimpleNamespace(
            name="edit_file",
            arguments="{}",
        ),
    )
    tool_message = SimpleNamespace(
        role="assistant",
        content="",
        tool_calls=[tool_call],
    )
    final_message = SimpleNamespace(
        role="assistant",
        content="了解しました。",
        tool_calls=[],
    )

    responses = [
        SimpleNamespace(choices=[SimpleNamespace(message=tool_message)]),
        SimpleNamespace(choices=[SimpleNamespace(message=final_message)]),
    ]

    monkeypatch.setattr(
        runtime_module,
        "ask_llm",
        lambda _messages, tools=None: responses.pop(0),
    )

    confirmations = []

    runtime = AgentRuntime(
        tmp_path,
        tool_registry=registry,
        confirm=lambda message: confirmations.append(message) or False,
    )

    result = runtime.run("ファイルを変更してください")

    assert result == "了解しました。"
    assert runtime.task is not None
    assert runtime.task.status.value == "completed"
    assert executed["value"] is False
    assert confirmations
    assert any(
        json.loads(message["content"]).get("user_rejected") is True
        for message in runtime.messages
        if message.get("role") == "tool"
    )


def test_runtime_records_max_iterations(
    tmp_path: Path,
    monkeypatch,
) -> None:
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="loop",
            description="Keep working",
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
            },
            handler=lambda _working_directory, _arguments: {"ok": True},
        )
    )

    tool_call = SimpleNamespace(
        id="call-loop",
        function=SimpleNamespace(
            name="loop",
            arguments="{}",
        ),
    )
    tool_message = SimpleNamespace(
        role="assistant",
        content="",
        tool_calls=[tool_call],
    )

    monkeypatch.setattr(
        runtime_module,
        "ask_llm",
        lambda _messages, tools=None: SimpleNamespace(
            choices=[SimpleNamespace(message=tool_message)]
        ),
    )

    runtime = AgentRuntime(
        tmp_path,
        max_iterations=2,
        tool_registry=registry,
    )

    result = runtime.run("終わらない作業")

    assert "最大反復回数" in result
    assert runtime.task is not None
    assert runtime.task.status.value == "max_iterations"
    assert runtime.task.iteration == 2


def test_runtime_tracks_multiple_tasks_with_independent_history(
    tmp_path: Path,
    monkeypatch,
) -> None:
    seen_messages = []
    responses = [
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        role="assistant",
                        content="first done",
                        tool_calls=[],
                    )
                )
            ]
        ),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        role="assistant",
                        content="second done",
                        tool_calls=[],
                    )
                )
            ]
        ),
    ]

    def fake_ask_llm(messages, tools=None):
        seen_messages.append(messages)
        return responses.pop(0)

    monkeypatch.setattr(runtime_module, "ask_llm", fake_ask_llm)

    runtime = AgentRuntime(tmp_path)

    assert runtime.run("first") == "first done"
    first_task = runtime.current_task
    assert runtime.run("second") == "second done"
    second_task = runtime.current_task

    tasks = runtime.list_tasks()

    assert first_task is None
    assert second_task is None
    assert tasks == []

    assert any(
        message.get("role") == "user"
        and message.get("content") == "first"
        for message in seen_messages[0]
    )
    assert any(
        message.get("role") == "user"
        and message.get("content") == "first"
        for message in seen_messages[1]
    )
    assert all(
        message.get("content") != "second"
        for message in seen_messages[0]
    )


def test_runtime_blocks_task_wide_duplicate_tool_calls(
    tmp_path: Path,
    monkeypatch,
) -> None:
    registry = ToolRegistry()
    executed = {"count": 0}

    def inspect(_working_directory, _arguments):
        executed["count"] += 1
        return {"ok": True, "value": "already inspected"}

    registry.register(
        ToolDefinition(
            name="inspect",
            description="Inspect something",
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
            },
            handler=inspect,
        )
    )

    def make_tool_response(name):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        role="assistant",
                        content="",
                        tool_calls=[
                            SimpleNamespace(
                                id=f"call-{name}-{executed['count']}",
                                function=SimpleNamespace(
                                    name=name,
                                    arguments="{}",
                                ),
                            )
                        ],
                    )
                )
            ]
        )

    final_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    role="assistant",
                    content="結果を確認しました。",
                    tool_calls=[],
                )
            )
        ]
    )

    responses = [
        make_tool_response("inspect"),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        role="assistant",
                        content="",
                        tool_calls=[
                            SimpleNamespace(
                                id="call-other",
                                function=SimpleNamespace(
                                    name="other",
                                    arguments="{}",
                                ),
                            )
                        ],
                    )
                )
            ]
        ),
        make_tool_response("inspect"),
        final_response,
    ]

    registry.register(
        ToolDefinition(
            name="other",
            description="Do another inspection",
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
            },
            handler=lambda _working_directory, _arguments: {
                "ok": True,
                "value": "different observation",
            },
        )
    )

    monkeypatch.setattr(
        runtime_module,
        "ask_llm",
        lambda _messages, tools=None: responses.pop(0),
    )

    runtime = AgentRuntime(
        tmp_path,
        tool_registry=registry,
    )

    result = runtime.run("調査してください")

    assert result == "結果を確認しました。"
    assert executed["count"] == 1
    assert any(
        json.loads(message["content"]).get("repeated_tool_call") is True
        for message in runtime.messages
        if message.get("role") == "tool"
    )


def test_runtime_retries_invalid_empty_final_response(
    tmp_path: Path,
    monkeypatch,
) -> None:
    invalid_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    role="assistant",
                    content="{}",
                    tool_calls=[],
                )
            )
        ]
    )
    final_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    role="assistant",
                    content="調査結果です。",
                    tool_calls=[],
                )
            )
        ]
    )
    responses = [invalid_response, final_response]

    monkeypatch.setattr(
        runtime_module,
        "ask_llm",
        lambda _messages, tools=None: responses.pop(0),
    )

    runtime = AgentRuntime(tmp_path)

    result = runtime.run("調査してください")

    assert result == "調査結果です。"
    assert runtime.task is not None
    assert runtime.task.status.value == "completed"


def test_runtime_rejects_blank_final_response() -> None:
    assert AgentRuntime._is_invalid_final_response("") is True
    assert AgentRuntime._is_invalid_final_response("  ") is True
    assert AgentRuntime._is_invalid_final_response("{}") is True
    assert AgentRuntime._is_invalid_final_response("[]") is True
    assert AgentRuntime._is_invalid_final_response("完了しました") is False


def test_runtime_retries_when_model_echoes_tool_result(
    tmp_path: Path,
    monkeypatch,
) -> None:
    tool_result = {"ok": True, "value": "observed"}
    tool_call = SimpleNamespace(
        id="call-1",
        function=SimpleNamespace(
            name="inspect",
            arguments="{}",
        ),
    )
    tool_message = SimpleNamespace(
        role="assistant",
        content="",
        tool_calls=[tool_call],
    )
    echoed_final = SimpleNamespace(
        role="assistant",
        content=json.dumps(tool_result, ensure_ascii=False, sort_keys=True),
        tool_calls=[],
    )
    real_final = SimpleNamespace(
        role="assistant",
        content="観測結果を確認しました。",
        tool_calls=[],
    )

    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="inspect",
            description="Inspect",
            parameters={"type": "object", "properties": {}, "required": []},
            handler=lambda _working_directory, _arguments: tool_result,
        )
    )

    responses = [
        SimpleNamespace(choices=[SimpleNamespace(message=tool_message)]),
        SimpleNamespace(choices=[SimpleNamespace(message=echoed_final)]),
        SimpleNamespace(choices=[SimpleNamespace(message=real_final)]),
    ]

    monkeypatch.setattr(
        runtime_module,
        "ask_llm",
        lambda _messages, tools=None: responses.pop(0),
    )

    runtime = AgentRuntime(
        tmp_path,
        tool_registry=registry,
    )

    result = runtime.run("調査してください")

    assert result == "観測結果を確認しました。"
    assert runtime.task is not None
    assert runtime.task.status.value == "completed"



def test_runtime_treats_read_ranges_with_same_content_as_same_observation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    registry = ToolRegistry()

    responses = []

    def make_tool_call(call_id: str, end_line: int | None) -> SimpleNamespace:
        arguments = {"path": "README.md"}
        if end_line is not None:
            arguments["start_line"] = 1
            arguments["end_line"] = end_line
        return SimpleNamespace(
            id=call_id,
            function=SimpleNamespace(
                name="read_file",
                arguments=json.dumps(arguments),
            ),
        )

    responses.extend(
        [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            role="assistant",
                            content="",
                            tool_calls=[make_tool_call("read-1", 20)],
                        )
                    )
                ]
            ),
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            role="assistant",
                            content="",
                            tool_calls=[make_tool_call("read-2", 40)],
                        )
                    )
                ]
            ),
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            role="assistant",
                            content="調査結果をまとめました。",
                            tool_calls=[],
                        )
                    )
                ]
            ),
        ]
    )

    registry.register(
        ToolDefinition(
            name="read_file",
            description="Read file",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start_line": {"type": "integer"},
                    "end_line": {"type": "integer"},
                },
                "required": ["path"],
            },
            handler=lambda _working_directory, _arguments: {
                "ok": True,
                "path": "README.md",
                "start_line": 1,
                "end_line": _arguments.get("end_line"),
                "content": "same useful content",
                "truncated": False,
            },
        )
    )

    monkeypatch.setattr(
        runtime_module,
        "ask_llm",
        lambda _messages, tools=None: responses.pop(0),
    )

    runtime = AgentRuntime(
        tmp_path,
        tool_registry=registry,
    )

    assert runtime.run("README.mdを調べてください") == "調査結果をまとめました。"
    assert runtime.task is not None
    assert runtime.task.observations[0].progress_state.value == "progressed"
    assert runtime.task.observations[1].progress_state.value == "no_progress"
    assert runtime.task.no_progress_streak == 1
    assert runtime.task.observations[0].new_information is True
    assert runtime.task.observations[1].new_information is False
    assert runtime.task.observations[1].progress_state.value == "no_progress"


def test_runtime_keeps_tool_quarantine_inside_task_state(
    tmp_path: Path,
    monkeypatch,
) -> None:
    registry = ToolRegistry()
    tool_call = SimpleNamespace(
        id="call-1",
        function=SimpleNamespace(
            name="inspect",
            arguments="{}",
        ),
    )
    final = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    role="assistant",
                    content="完了しました。",
                    tool_calls=[],
                )
            )
        ]
    )
    responses = [
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        role="assistant",
                        content="",
                        tool_calls=[tool_call],
                    )
                )
            ]
        ),
        final,
    ]
    registry.register(
        ToolDefinition(
            name="inspect",
            description="Inspect",
            parameters={"type": "object", "properties": {}, "required": []},
            handler=lambda _working_directory, _arguments: {
                "ok": True,
                "value": "done",
            },
        )
    )
    monkeypatch.setattr(
        runtime_module,
        "ask_llm",
        lambda _messages, tools=None: responses.pop(0),
    )

    runtime = AgentRuntime(tmp_path, tool_registry=registry, confirm=lambda _message: True)
    assert runtime.run("調査") == "完了しました。"
    assert runtime.task is not None
    assert runtime.task.disabled_tools == set()


def test_runtime_normalizes_message_like_final_content() -> None:
    message_like = {
        "role": "assistant",
        "content": "こんにちは！",
        "tool_calls": [],
    }

    assert AgentRuntime._normalize_final_content(message_like) == "こんにちは！"
    assert AgentRuntime._normalize_final_content("  直接回答  ") == "直接回答"


def test_runtime_normalizes_json_encoded_message_content() -> None:
    encoded = (
        '{"role":"assistant",'
        '"content":"こんにちは！何かお手伝いできますか？",'
        '"tool_calls":[]}'
    )

    assert (
        AgentRuntime._normalize_final_content(encoded)
        == "こんにちは！何かお手伝いできますか？"
    )


def test_runtime_exposes_web_tools_for_live_information(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured = []

    final = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    role="assistant",
                    content="現在の天気情報を取得します。",
                    tool_calls=[],
                )
            )
        ]
    )

    def fake_ask_llm(messages, tools=None):
        captured.append(tools or [])
        return final

    monkeypatch.setattr(runtime_module, "ask_llm", fake_ask_llm)

    runtime = AgentRuntime(tmp_path)
    result = runtime.run("今日の天気は？")

    assert result == "現在の天気情報を取得します。"
    names = {tool["function"]["name"] for tool in captured[0]}
    assert "search_web" in names
    assert "fetch_web_page" in names
    assert "finish_task" in names


def test_runtime_keeps_task_history_isolated_while_sharing_conversation_context(
    tmp_path: Path,
    monkeypatch,
) -> None:
    responses = [
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        role="assistant",
                        content="了解しました。READMEを確認します。",
                        tool_calls=[],
                    )
                )
            ]
        ),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        role="assistant",
                        content="先ほどの話を踏まえて続けます。",
                        tool_calls=[],
                    )
                )
            ]
        ),
    ]
    seen = []

    monkeypatch.setattr(
        runtime_module,
        "ask_llm",
        lambda messages, tools=None: (
            seen.append(messages) or responses.pop(0)
        ),
    )

    runtime = AgentRuntime(tmp_path)

    assert runtime.run("最初の話をしたい") == "了解しました。READMEを確認します。"
    first = runtime.current_task
    assert runtime.run("先ほどの話を踏まえて続けて") == "先ほどの話を踏まえて続けます。"
    second = runtime.current_task

    assert first is None
    assert second is None
    assert any(
        message.get("content") == "最初の話をしたい"
        for message in seen[1]
        if message.get("role") == "user"
    )
    assert any(
        message.get("content") == "了解しました。READMEを確認します。"
        for message in seen[1]
        if message.get("role") == "assistant"
    )


def test_runtime_quarantines_failed_tool_for_next_recovery_step(
    tmp_path: Path,
    monkeypatch,
) -> None:
    registry = ToolRegistry()
    executed = {"count": 0}

    def failing_tool(_working_directory, _arguments):
        executed["count"] += 1
        return {"ok": False, "error": "command failed"}

    registry.register(
        ToolDefinition(
            name="run_action",
            description="Run an action",
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
            },
            handler=failing_tool,
        )
    )

    def tool_call(call_id: str):
        return SimpleNamespace(
            id=call_id,
            function=SimpleNamespace(
                name="run_action",
                arguments="{}",
            ),
        )

    responses = [
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        role="assistant",
                        content="",
                        tool_calls=[tool_call("call-1")],
                    )
                )
            ]
        ),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        role="assistant",
                        content="",
                        tool_calls=[tool_call("call-2")],
                    )
                )
            ]
        ),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        role="assistant",
                        content="原因を確認できないため、ここで止めます。",
                        tool_calls=[],
                    )
                )
            ]
        ),
    ]

    captured_tools = []

    def fake_ask_llm(messages, tools=None):
        captured_tools.append(tools or [])
        return responses.pop(0)

    monkeypatch.setattr(runtime_module, "ask_llm", fake_ask_llm)

    runtime = AgentRuntime(tmp_path, tool_registry=registry)

    assert runtime.run("アクションを実行してください") == (
        "原因を確認できないため、ここで止めます。"
    )
    assert executed["count"] == 1
    assert runtime.task is not None
    assert runtime.task.recovery_tool == "run_action"
    assert all(
        schema["function"]["name"] != "run_action"
        for schema in captured_tools[1]
    )
    assert any(
        json.loads(message["content"]).get("recovery_blocked") is True
        for message in runtime.messages
        if message.get("role") == "tool"
    )


def test_runtime_retries_scoped_request_when_model_only_explains(
    tmp_path: Path,
    monkeypatch,
) -> None:
    tool_call = SimpleNamespace(
        id="call-edit",
        function=SimpleNamespace(
            name="file_mutation",
            arguments=json.dumps(
                {
                    "operation": "edit",
                    "path": "test.txt",
                    "search_text": "old",
                    "replace_text": "new",
                }
            ),
        ),
    )
    responses = [
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        role="assistant",
                        content="以下のように編集してください。",
                        tool_calls=[],
                    )
                )
            ]
        ),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        role="assistant",
                        content="",
                        tool_calls=[tool_call],
                    )
                )
            ]
        ),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        role="assistant",
                        content="編集を実行しました。",
                        tool_calls=[],
                    )
                )
            ]
        ),
    ]

    (tmp_path / "test.txt").write_text("old", encoding="utf-8")

    monkeypatch.setattr(
        runtime_module,
        "ask_llm",
        lambda _messages, tools=None: responses.pop(0),
    )

    runtime = AgentRuntime(
        tmp_path,
        confirm=lambda _message: True,
    )

    assert runtime.run("test.txtを修正してください") == "編集を実行しました。"
    assert (tmp_path / "test.txt").read_text(encoding="utf-8") == "new"
    assert runtime.task is not None
    assert runtime.task.tool_calls == 1


def test_runtime_synthesizes_immediately_after_terminal_tool_success(
    tmp_path: Path,
    monkeypatch,
) -> None:
    registry = ToolRegistry()
    executed = {"count": 0}

    def create(_working_directory, _arguments):
        executed["count"] += 1
        return {"ok": True, "created": True, "path": "test.html"}

    registry.register(
        ToolDefinition(
            name="create_file",
            description="Create a file",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
            handler=create,
            terminal_on_success=True,
        )
    )

    tool_call = SimpleNamespace(
        id="call-create",
        function=SimpleNamespace(
            name="create_file",
            arguments=json.dumps(
                {"path": "test.html", "content": "<html></html>"}
            ),
        ),
    )
    tool_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    role="assistant",
                    content="",
                    tool_calls=[tool_call],
                )
            )
        ]
    )
    final_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    role="assistant",
                    content="HTMLファイルを作成しました。",
                    tool_calls=[],
                )
            )
        ]
    )

    captured_tools = []
    responses = [tool_response, final_response]

    def fake_ask_llm(_messages, tools=None):
        captured_tools.append(tools or [])
        return responses.pop(0)

    monkeypatch.setattr(runtime_module, "ask_llm", fake_ask_llm)

    runtime = AgentRuntime(
        tmp_path,
        tool_registry=registry,
    )

    assert runtime.run("HTMLファイルを作成してください") == (
        "HTMLファイルを作成しました。"
    )
    assert executed["count"] == 1
    assert len(captured_tools) == 2
    assert captured_tools[0]
    assert captured_tools[1] == []




def test_runtime_finishes_without_extra_llm_synthesis_after_finish_task(
    tmp_path: Path,
    monkeypatch,
) -> None:
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="inspect",
            description="Inspect",
            parameters={"type": "object", "properties": {}, "required": []},
            handler=lambda _working_directory, _arguments: {
                "ok": True,
                "value": "verified",
            },
        )
    )
    registry.register(
        ToolDefinition(
            name="finish_task",
            description="Finish",
            parameters={
                "type": "object",
                "properties": {
                    "completion_status": {"type": "string", "enum": ["completed", "blocked"]},
                    "summary": {"type": "string"},
                },
                "required": ["completion_status", "summary"],
            },
            handler=lambda _working_directory, arguments: {
                "ok": True,
                "completion_status": arguments["completion_status"],
                "summary": arguments["summary"],
            },
        )
    )

    responses = [
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        role="assistant",
                        content="",
                        tool_calls=[
                            SimpleNamespace(
                                id="inspect-1",
                                function=SimpleNamespace(
                                    name="inspect",
                                    arguments="{}",
                                ),
                            )
                        ],
                    )
                )
            ]
        ),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        role="assistant",
                        content="",
                        tool_calls=[
                            SimpleNamespace(
                                id="finish-1",
                                function=SimpleNamespace(
                                    name="finish_task",
                                    arguments='{"completion_status":"completed","summary":"検証して完了しました。"}',
                                ),
                            )
                        ],
                    )
                )
            ]
        ),
    ]

    monkeypatch.setattr(
        runtime_module,
        "ask_llm",
        lambda _messages, tools=None: responses.pop(0),
    )

    runtime = AgentRuntime(tmp_path, tool_registry=registry)
    assert runtime.run("調査して完了してください。") == "検証して完了しました。"
    assert runtime.task is not None
    assert runtime.task.status.value == "completed"
    assert runtime.task.iteration == 2


def test_runtime_reuses_environment_snapshot_between_unchanged_iterations(
    tmp_path: Path,
    monkeypatch,
) -> None:
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="inspect",
            description="Inspect",
            parameters={"type": "object", "properties": {}, "required": []},
            handler=lambda _working_directory, _arguments: {
                "ok": True,
                "value": "verified",
            },
        )
    )
    responses = [
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        role="assistant",
                        content="",
                        tool_calls=[
                            SimpleNamespace(
                                id="inspect-1",
                                function=SimpleNamespace(
                                    name="inspect",
                                    arguments="{}",
                                ),
                            )
                        ],
                    )
                )
            ]
        ),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        role="assistant",
                        content="確認結果です。",
                        tool_calls=[],
                    )
                )
            ]
        ),
    ]
    calls = {"count": 0}

    def fake_environment(_workspace, _paths=()):
        calls["count"] += 1
        return "[Environment]\nWorkspace: test"

    monkeypatch.setattr(runtime_module, "build_environment_context", fake_environment)
    monkeypatch.setattr(
        runtime_module,
        "ask_llm",
        lambda _messages, tools=None: responses.pop(0),
    )

    runtime = AgentRuntime(tmp_path, tool_registry=registry)
    assert runtime.run("調査してください") == "確認結果です。"
    assert calls["count"] == 1


def test_system_prompt_is_compact_but_retains_core_agent_rules() -> None:
    prompt = runtime_module.SYSTEM_PROMPT

    assert len(prompt) < 2600
    for phrase in ("Tool", "Recovery Guide", "finish_task", "ask_user", "Session Context", "Web"):
        assert phrase in prompt


def test_runtime_recovers_when_model_emits_raw_tool_call_markup(
    tmp_path: Path,
    monkeypatch,
) -> None:
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="run_python_script",
            description="Run a Python script",
            parameters={
                "type": "object",
                "properties": {"script": {"type": "string"}},
                "required": ["script"],
            },
            handler=lambda _working_directory, arguments: {
                "ok": True,
                "stdout": "ok",
                "script": arguments["script"],
            },
        )
    )

    raw_tool_call = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    role="assistant",
                    content='<|tool_call>call:run_python_script{"script":"print(1)"}<|tool_call>',
                    tool_calls=[],
                )
            )
        ]
    )
    structured_tool_call = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    role="assistant",
                    content="",
                    tool_calls=[
                        SimpleNamespace(
                            id="call-python",
                            function=SimpleNamespace(
                                name="run_python_script",
                                arguments='{"script":"print(1)"}',
                            ),
                        )
                    ],
                )
            )
        ]
    )
    final = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    role="assistant",
                    content="テスト実行が完了しました。",
                    tool_calls=[],
                )
            )
        ]
    )

    responses = [raw_tool_call, structured_tool_call, final]
    monkeypatch.setattr(
        runtime_module,
        "ask_llm",
        lambda _messages, tools=None: responses.pop(0),
    )

    runtime = AgentRuntime(
        tmp_path,
        tool_registry=registry,
        confirm=lambda _message: True,
    )

    assert runtime.run("Pythonを実行してください") == "テスト実行が完了しました。"
    assert runtime.task is not None
    assert runtime.task.tool_calls == 1


def test_runtime_allows_verification_after_state_change(
    tmp_path: Path,
    monkeypatch,
) -> None:
    registry = ToolRegistry()
    calls = {"read": 0}

    def read_file(_working_directory, _arguments):
        calls["read"] += 1
        return {
            "ok": True,
            "content": "new" if calls["read"] > 1 else "old",
        }

    registry.register(
        ToolDefinition(
            name="read_file",
            description="Read a file",
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
            handler=read_file,
        )
    )
    registry.register(
        ToolDefinition(
            name="file_mutation",
            description="Mutate a file",
            parameters={
                "type": "object",
                "properties": {
                    "operation": {"type": "string"},
                    "path": {"type": "string"},
                },
                "required": ["operation", "path"],
            },
            handler=lambda _working_directory, _arguments: {
                "ok": True,
                "operation": "edit",
                "path": "test.txt",
            },
        )
    )

    def tool_call(call_id: str, name: str, arguments: str) -> SimpleNamespace:
        return SimpleNamespace(
            id=call_id,
            function=SimpleNamespace(name=name, arguments=arguments),
        )

    responses = [
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        role="assistant",
                        content="",
                        tool_calls=[
                            tool_call("read-1", "read_file", '{"path":"test.txt"}')
                        ],
                    )
                )
            ]
        ),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        role="assistant",
                        content="",
                        tool_calls=[
                            tool_call("edit-1", "file_mutation", '{"operation":"edit","path":"test.txt"}')
                        ],
                    )
                )
            ]
        ),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        role="assistant",
                        content="",
                        tool_calls=[
                            tool_call("read-2", "read_file", '{"path":"test.txt"}')
                        ],
                    )
                )
            ]
        ),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        role="assistant",
                        content="変更後の内容を確認しました。",
                        tool_calls=[],
                    )
                )
            ]
        ),
    ]

    monkeypatch.setattr(
        runtime_module,
        "ask_llm",
        lambda _messages, tools=None: responses.pop(0),
    )

    runtime = AgentRuntime(tmp_path, tool_registry=registry, confirm=lambda _message: True)

    assert runtime.run("test.txtを変更して内容を確認してください") == "変更後の内容を確認しました。"
    assert runtime.task is not None
    assert runtime.task.tool_calls == 3
    assert calls["read"] == 2
