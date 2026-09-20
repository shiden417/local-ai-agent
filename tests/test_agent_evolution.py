from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from agent.capabilities import Capability
from agent.request_classifier import RequestClassifier, RequestMode
from agent.environment import build_environment_context
from agent.runtime import AgentRuntime
from agent.session import SessionManager
from agent.task import TaskState
from agent.tool_registry import ToolDefinition, ToolRegistry
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
    assert runtime.session_manager.recent_conversation_messages()[-1]["content"] == answer


def test_request_classifier_separates_obvious_conversation_from_agent_tasks():
    classifier = RequestClassifier()

    assert classifier.classify("こんにちは").mode == RequestMode.DIRECT
    assert classifier.classify("明日の天気を教えて").mode == RequestMode.TASK


def test_agent_tasks_receive_registered_tools_for_model_selection():
    registry = create_default_tool_registry()

    names = {
        item["function"]["name"]
        for item in registry.schemas_for(include_control_tools=True)
    }

    assert {"search_web", "fetch_web_page", "ask_user", "finish_task"} <= names
    assert "run_python_script" in names


def test_session_context_keeps_compact_web_facts():
    context = SessionManager()
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
    monkeypatch.setattr(
        fetch_module.socket,
        "getaddrinfo",
        lambda *args, **kwargs: [
            (2, 1, 6, "", ("93.184.216.34", 443)),
        ],
    )

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
    error = runtime.completion_verifier.verify(
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

    assert runtime.completion_verifier.verify(
        task,
        {"completion_status": "completed"},
    ) is None



def test_session_context_follow_up_reuses_previous_capability(tmp_path: Path) -> None:
    runtime = AgentRuntime(tmp_path)
    runtime.session_manager.remember_task(
        "Python 3.14について調べて",
        "Python 3.14の変更点を確認しました。",
        [
            {
                "role": "tool",
                "name": "search_web",
                "content": (
                    '{"ok":true,"count":1,"results":['
                    '{"title":"Python 3.14","url":"https://python.org",'
                    '"snippet":"Official release information"}]}'
                ),
            }
        ],
    )

    routing_text = runtime._routing_text("その中で重要な変更を3つ教えて")

    assert "Python 3.14について調べて" in routing_text
    assert "その中で重要な変更を3つ教えて" in routing_text


def test_stale_previous_answer_is_rejected_after_new_observation(tmp_path: Path) -> None:
    runtime = AgentRuntime(tmp_path)
    runtime.session_manager.last_answer = "前のTaskの回答です。"
    task = runtime.task_manager.create("現在のTaskを調べる")
    task.messages = [
        {"role": "user", "content": task.goal},
        {
            "role": "tool",
            "name": "list_directory",
            "content": '{"ok":true,"entries":["new.txt"]}',
        },
    ]

    assert runtime._is_stale_session_response(
        "前のTaskの回答です。",
        task,
    ) is True


def test_previous_answer_without_current_observation_is_not_treated_as_stale(
    tmp_path: Path,
) -> None:
    runtime = AgentRuntime(tmp_path)
    runtime.session_manager.last_answer = "同じ説明です。"
    task = runtime.task_manager.create("会話する")

    assert runtime._is_stale_session_response("同じ説明です。", task) is False


def test_latest_web_completion_requires_page_fetch(tmp_path: Path) -> None:
    runtime = AgentRuntime(tmp_path)
    task = TaskState("Python 3.14の最新情報を調査してください")
    task.messages = [
        {
            "role": "tool",
            "name": "search_web",
            "content": (
                '{"ok":true,"count":2,"results":['
                '{"title":"result","url":"https://example.com","snippet":"latest"}]}'
            ),
        }
    ]

    error = runtime.completion_verifier.verify(
        task,
        {"completion_status": "completed"},
    )

    assert error is not None
    assert "fetch" in error.lower()


def test_latest_web_completion_accepts_fetched_source(tmp_path: Path) -> None:
    runtime = AgentRuntime(tmp_path)
    task = TaskState("Python 3.14の公式リリース情報を確認")
    task.messages = [
        {
            "role": "tool",
            "name": "search_web",
            "content": (
                '{"ok":true,"count":1,"results":['
                '{"title":"release","url":"https://python.org","snippet":"release"}]}'
            ),
        },
        {
            "role": "tool",
            "name": "fetch_web_page",
            "content": (
                '{"ok":true,"status_code":200,"title":"Release",'
                '"content":"official release information"}'
            ),
        },
    ]

    assert runtime.completion_verifier.verify(
        task,
        {"completion_status": "completed"},
    ) is None



def test_project_investigation_cannot_finish_from_listing_only(tmp_path: Path) -> None:
    runtime = AgentRuntime(tmp_path)
    task = TaskState(
        "このプロジェクトの現在の状態を確認して、必要なら問題点を調査してください"
    )
    task.messages = [
        {
            "role": "tool",
            "name": "list_directory",
            "content": '{"ok":true,"entries":["agent","tests"]}',
        }
    ]

    error = runtime.completion_verifier.verify(
        task,
        {"completion_status": "completed"},
    )

    assert error is not None
    assert "diagnostic" in error.lower()


def test_project_investigation_accepts_concrete_diagnostic(tmp_path: Path) -> None:
    runtime = AgentRuntime(tmp_path)
    task = TaskState(
        "このプロジェクトの現在の状態を確認して、必要なら問題点を調査してください"
    )
    task.messages = [
        {
            "role": "tool",
            "name": "list_directory",
            "content": '{"ok":true,"entries":["agent","tests"]}',
        },
        {
            "role": "tool",
            "name": "read_file",
            "content": '{"ok":true,"path":"README.md","content":"state"}',
        },
    ]

    assert runtime.completion_verifier.verify(
        task,
        {"completion_status": "completed"},
    ) is None



def test_fetch_web_page_blocks_localhost() -> None:
    result = fetch_module.fetch_web_page("http://localhost:8080/internal")

    assert result["ok"] is False
    assert "localhost" in result["error"].lower()


def test_fetch_web_page_blocks_private_ip() -> None:
    result = fetch_module.fetch_web_page("http://192.168.1.1/status")

    assert result["ok"] is False
    assert "private" in result["error"].lower()



def test_session_context_preserves_facts_for_follow_up(tmp_path: Path) -> None:
    context = SessionManager()
    context.remember_task(
        "Python 3.14について調べて",
        "公式情報を確認しました。",
        [
            {
                "role": "tool",
                "name": "search_web",
                "content": (
                    '{"ok":true,"results":['
                    '{"title":"Python 3.14","url":"https://python.org",'
                    '"snippet":"Official release"}]}'
                ),
            }
        ],
    )
    context.remember_task(
        "その中で重要な変更を3つ教えて",
        "重要な変更を3つまとめました。",
        [],
    )

    prompt = context.prompt_block()
    assert "Python 3.14" in prompt
    assert "https://python.org" in prompt



def test_session_context_keeps_topic_anchor_across_multiple_follow_ups(
    tmp_path: Path,
) -> None:
    runtime = AgentRuntime(tmp_path)
    runtime.session_manager.remember_task(
        "Python 3.14について調べて",
        "調査しました。",
        [],
    )
    runtime.session_manager.remember_task(
        "その中で重要な変更を3つ教えて",
        "3つまとめました。",
        [],
    )

    routing_text = runtime._routing_text("その3つのうち開発で重要なものは？")

    assert "Python 3.14について調べて" in routing_text
    assert "その3つのうち開発で重要なものは？" in routing_text



def test_completed_task_does_not_pollute_conversation_history(
    tmp_path: Path,
    monkeypatch,
) -> None:
    registry = ToolRegistry()

    def run_action(_working_directory, _arguments):
        return {"ok": True, "message": "done"}

    registry.register(
        ToolDefinition(
            name="run_action",
            description="Run an action",
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
            },
            handler=run_action,
            capabilities=(Capability.PROCESS,),
        )
    )

    tool_call = SimpleNamespace(
        id="call-1",
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
                        content="",
                        tool_calls=[tool_call],
                    )
                )
            ]
        ),
        _llm_response("調査結果を確認しました。"),
    ]
    monkeypatch.setattr(
        "agent.runtime.ask_llm",
        lambda messages, tools=None: responses.pop(0),
    )

    runtime = AgentRuntime(tmp_path, tool_registry=registry)
    assert runtime.run("アクションを実行してください") == "調査結果を確認しました。"

    assert runtime.session_manager.recent_conversation_messages() == []
    assert runtime.session_manager.last_answer == "調査結果を確認しました。"



def test_clear_session_context_resets_ephemeral_context(tmp_path: Path) -> None:
    runtime = AgentRuntime(tmp_path)
    runtime.session_manager.remember_task("topic", "answer", [])
    runtime.session_manager.add_conversation_turn("hello", "world")

    runtime.clear_session_context()

    assert runtime.session_manager.has_context is False
    assert runtime.session_manager.recent_conversation_messages() == []
