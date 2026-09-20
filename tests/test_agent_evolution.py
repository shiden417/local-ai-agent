from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from agent.capability_router import Capability, CapabilityRouter, RoutingMode
from agent.environment import build_environment_context
from agent.runtime import AgentRuntime
from agent.session_context import SessionContext
from agent.task import TaskState
from agent.tool_registry import ToolRegistry
from agent.tools import create_default_tool_registry
import tools.fetch_web_page as fetch_module


def _llm_response(content: str):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=content,
                    tool_calls=[],
                )
            )
        ]
    )


def test_direct_conversation_does_not_create_task(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        "agent.runtime.ask_llm",
        lambda messages, tools=None: _llm_response("こんにちは。どうしましたか？"),
    )

    runtime = AgentRuntime(tmp_path)
    answer = runtime.run("こんにちは")

    assert answer == "こんにちは。どうしましたか？"
    assert runtime.list_tasks() == []
    assert runtime.conversation_manager.recent_messages()[-1]["content"] == answer


def test_control_tools_are_scoped_to_agent_tasks():
    registry = create_default_tool_registry()
    router = CapabilityRouter()

    assert router.route("こんにちは").mode == RoutingMode.DIRECT
    assert registry.schemas_for("こんにちは") == []

    scoped = registry.schemas_for(
        "明日の天気を教えて",
        include_control_tools=True,
    )
    names = {
        item["function"]["name"]
        for item in scoped
    }
    assert {"search_web", "fetch_web_page", "ask_user", "finish_task"} <= names


def test_future_weather_is_action_routed():
    route = CapabilityRouter().route("明日の東京の天気を教えて")
    assert route.mode == RoutingMode.SCOPED
    assert Capability.WEB_SEARCH in route.capabilities


def test_session_context_keeps_compact_web_facts():
    context = SessionContext()
    context.remember_task(
        "東京の明日の天気を調べて",
        "明日は晴れで、最高気温は28℃です。",
        [
            {
                "role": "tool",
                "name": "search_web",
                "content": (
                    '{"ok":true,"count":1,"results":['
                    '{"title":"天気予報","url":"https://example.com/weather",'
                    '"snippet":"晴れ、最高気温28℃"}]}'
                ),
            }
        ],
    )

    prompt = context.prompt_block()
    assert "東京の明日の天気を調べて" in prompt
    assert "最高気温28" in prompt
    assert "https://example.com/weather" in prompt
    assert "results" not in prompt


def test_environment_context_loads_hierarchical_agents(tmp_path: Path):
    root_agents = tmp_path / "AGENTS.md"
    src = tmp_path / "src"
    src.mkdir()
    local_agents = src / "AGENTS.md"
    root_agents.write_text("Root rule\n", encoding="utf-8")
    local_agents.write_text("Local rule\n", encoding="utf-8")

    context = build_environment_context(tmp_path, ["src/app.py"])

    assert "Root rule" in context
    assert "Local rule" in context
    assert context.index("Root rule") < context.index("Local rule")
    assert "Workspace root entries:" in context


def test_fetch_web_page_extracts_readable_html(monkeypatch):
    class FakeHeaders:
        def get_content_type(self):
            return "text/html"
        def get_content_charset(self):
            return "utf-8"

    class FakeResponse:
        status = 200
        headers = FakeHeaders()

        def geturl(self):
            return "https://example.com/final"

        def read(self, limit):
            return b"<html><head><title>Example</title></head><body><h1>Hello</h1><p>World</p><script>x()</script></body></html>"

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(fetch_module, "urlopen", lambda request, timeout: FakeResponse())

    result = fetch_module.fetch_web_page("https://example.com")

    assert result["ok"] is True
    assert result["title"] == "Example"
    assert "Hello" in result["content"]
    assert "World" in result["content"]
    assert "x()" not in result["content"]


def test_finish_task_requires_a_successful_prior_action(tmp_path: Path):
    runtime = AgentRuntime(tmp_path)
    task = TaskState("ファイルを作成して")
    error = runtime._verify_finish_task(
        task,
        {"completion_status": "completed"},
    )

    assert error is not None
    assert "no successful action" in error.lower()


def test_finish_task_allows_successful_command(tmp_path: Path):
    runtime = AgentRuntime(tmp_path)
    task = TaskState("テストする")
    task.messages = [
        {
            "role": "tool",
            "name": "execute_command",
            "content": '{"ok":true,"exit_code":0,"stdout":"ok","stderr":""}',
        }
    ]

    assert runtime._verify_finish_task(
        task,
        {"completion_status": "completed"},
    ) is None
