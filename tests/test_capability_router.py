from agent.capability_router import Capability, CapabilityRouter, RoutingMode


def test_router_marks_clear_greeting_as_direct() -> None:
    route = CapabilityRouter().route("こんにちは")

    assert route.mode == RoutingMode.DIRECT
    assert route.capabilities == frozenset()


def test_router_scopes_workspace_tasks() -> None:
    route = CapabilityRouter().route(
        "このフォルダに何があるか調べてください"
    )

    assert route.mode == RoutingMode.SCOPED
    assert Capability.WORKSPACE_READ in route.capabilities


def test_router_scopes_file_editing_tasks() -> None:
    route = CapabilityRouter().route("READMEを修正してください")

    assert route.mode == RoutingMode.SCOPED
    assert Capability.WORKSPACE_READ in route.capabilities
    assert Capability.WORKSPACE_WRITE in route.capabilities


def test_router_scopes_process_tasks() -> None:
    route = CapabilityRouter().route("pytestを実行してください")

    assert route.mode == RoutingMode.SCOPED
    assert Capability.PROCESS in route.capabilities


def test_router_detects_current_weather_as_web_search() -> None:
    route = CapabilityRouter().route("今日の天気を教えてください")

    assert route.mode == RoutingMode.SCOPED
    assert Capability.WEB_SEARCH in route.capabilities


def test_router_detects_current_price_as_web_search() -> None:
    route = CapabilityRouter().route("現在の価格を調べてください")

    assert route.mode == RoutingMode.SCOPED
    assert route.capabilities == frozenset({Capability.WEB_SEARCH})


def test_router_scopes_memory_tasks() -> None:
    route = CapabilityRouter().route("前回の記憶を確認してください")

    assert route.mode == RoutingMode.SCOPED
    assert Capability.MEMORY_READ in route.capabilities


def test_router_keeps_ambiguous_requests_open_for_llm_choice() -> None:
    route = CapabilityRouter().route("どうすればよいですか")

    assert route.mode == RoutingMode.OPEN


def test_router_does_not_treat_vague_test_as_process_request() -> None:
    route = CapabilityRouter().route("テスト")

    assert route.mode == RoutingMode.OPEN
    assert route.capabilities == frozenset()


def test_router_detects_action_execution_request() -> None:
    route = CapabilityRouter().route("アクションを実行してください")

    assert route.mode == RoutingMode.SCOPED
    assert Capability.PROCESS in route.capabilities


def test_router_detects_explicit_test_execution() -> None:
    route = CapabilityRouter().route("pytestを実行してください")

    assert route.mode == RoutingMode.SCOPED
    assert route.capabilities == frozenset({Capability.PROCESS})


def test_router_scopes_file_creation_tasks() -> None:
    route = CapabilityRouter().route(
        r"C:\Users\example\TestProgramingにHTMLファイルを作成して"
    )

    assert route.mode == RoutingMode.SCOPED
    assert Capability.WORKSPACE_WRITE in route.capabilities
    assert Capability.WORKSPACE_READ in route.capabilities


def test_router_sends_topic_research_to_web_search() -> None:
    route = CapabilityRouter().route("Python 3.14について調べて")

    assert route.mode == RoutingMode.SCOPED
    assert Capability.WEB_SEARCH in route.capabilities
    assert Capability.WORKSPACE_WRITE not in route.capabilities


def test_router_does_not_treat_explanatory_changes_as_file_edits() -> None:
    route = CapabilityRouter().route("その中で特に重要な変更を3つ教えて")

    assert Capability.WORKSPACE_WRITE not in route.capabilities


def test_router_keeps_generic_workspace_research_local() -> None:
    route = CapabilityRouter().route("README.mdを調べてください")

    assert route.mode == RoutingMode.SCOPED
    assert Capability.WORKSPACE_READ in route.capabilities





def test_router_adds_workspace_read_to_project_investigation() -> None:
    route = CapabilityRouter().route(
        "このプロジェクトの現在の状態を確認して、必要なら問題点を調査してください"
    )

    assert route.mode == RoutingMode.SCOPED
    assert Capability.PROCESS in route.capabilities
    assert Capability.WORKSPACE_READ in route.capabilities



def test_router_scopes_project_inspection() -> None:
    route = CapabilityRouter().route("このプロジェクトを見て")

    assert route.mode == RoutingMode.SCOPED
    assert Capability.WORKSPACE_READ in route.capabilities


def test_router_scopes_bug_fix_without_requiring_a_filename() -> None:
    route = CapabilityRouter().route("このバグを修正して")

    assert route.mode == RoutingMode.SCOPED
    assert Capability.WORKSPACE_WRITE in route.capabilities
    assert Capability.WORKSPACE_READ in route.capabilities



def test_router_routes_official_release_research_to_web() -> None:
    route = CapabilityRouter().route(
        "Python 3.14の公式リリース情報を確認して、主な変更点を教えて"
    )

    assert route.mode == RoutingMode.SCOPED
    assert Capability.WEB_SEARCH in route.capabilities
    assert Capability.WORKSPACE_WRITE not in route.capabilities
