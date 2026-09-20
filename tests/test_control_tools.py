import json
from pathlib import Path
from types import SimpleNamespace

from agent import runtime as runtime_module
from agent.runtime import AgentRuntime


def _tool_call(name: str, arguments: dict, call_id: str = "call-1"):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(
            name=name,
            arguments=json.dumps(arguments),
        ),
    )


def test_runtime_finishes_explicitly(tmp_path: Path, monkeypatch) -> None:
    responses = [
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        role="assistant",
                        content="",
                        tool_calls=[
                            _tool_call(
                                "finish_task",
                                {
                                    "completion_status": "completed",
                                    "summary": "作業目的を満たしました。",
                                },
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
                        content="完了しました。",
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

    runtime = AgentRuntime(tmp_path)
    assert runtime.run("作業を完了してください") == "完了しました。"
    assert runtime.task is not None
    assert runtime.task.status.value == "completed"


def test_runtime_asks_user_and_returns_answer_to_model(
    tmp_path: Path,
    monkeypatch,
) -> None:
    seen = []
    responses = [
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        role="assistant",
                        content="",
                        tool_calls=[
                            _tool_call(
                                "ask_user",
                                {"question": "どちらを使用しますか？"},
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
                        content="Aを使用します。",
                        tool_calls=[],
                    )
                )
            ]
        ),
    ]

    def fake_ask_llm(messages, tools=None):
        seen.append(messages)
        return responses.pop(0)

    monkeypatch.setattr(runtime_module, "ask_llm", fake_ask_llm)

    runtime = AgentRuntime(
        tmp_path,
        ask_user=lambda question: "A" if "どちら" in question else "",
    )

    assert runtime.run("設定について確認してください") == "Aを使用します。"
    assert any(
        '"answer": "A"' in str(message.get("content", ""))
        for message in runtime.messages
        if message.get("role") == "tool"
    )
