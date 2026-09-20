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
        "edit_file",
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
    )

    result = runtime.run("挨拶してください")

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


def test_runtime_tracks_multiple_tasks(
    tmp_path: Path,
    monkeypatch,
) -> None:
    first = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    role="assistant",
                    content="first done",
                    tool_calls=[],
                )
            )
        ]
    )
    second = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    role="assistant",
                    content="second done",
                    tool_calls=[],
                )
            )
        ]
    )
    responses = [first, second]

    monkeypatch.setattr(
        runtime_module,
        "ask_llm",
        lambda _messages, tools=None: responses.pop(0),
    )

    runtime = AgentRuntime(tmp_path)

    assert runtime.run("first") == "first done"
    assert runtime.run("second") == "second done"

    tasks = runtime.list_tasks()

    assert len(tasks) == 2
    assert tasks[0].goal == "second"
    assert tasks[0].status.value == "completed"
    assert tasks[1].goal == "first"
    assert tasks[1].status.value == "completed"


def test_runtime_blocks_consecutive_duplicate_tool_calls(
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

    def make_tool_response():
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        role="assistant",
                        content="",
                        tool_calls=[
                            SimpleNamespace(
                                id="call-inspect",
                                function=SimpleNamespace(
                                    name="inspect",
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
        make_tool_response(),
        make_tool_response(),
        final_response,
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

    assert result == "結果を確認しました。"
    assert executed["count"] == 1
    assert any(
        json.loads(message["content"]).get("repeated_tool_call") is True
        for message in runtime.messages
        if message.get("role") == "tool"
    )
