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
    assert route.capabilities == frozenset({Capability.WORKSPACE_READ})


def test_router_scopes_file_editing_tasks() -> None:
    route = CapabilityRouter().route("READMEを修正してください")

    assert route.mode == RoutingMode.SCOPED
    assert Capability.WORKSPACE_READ in route.capabilities
    assert Capability.WORKSPACE_WRITE in route.capabilities


def test_router_scopes_process_tasks() -> None:
    route = CapabilityRouter().route("pytestを実行してください")

    assert route.mode == RoutingMode.SCOPED
    assert route.capabilities == frozenset({Capability.PROCESS})


def test_router_scopes_memory_tasks() -> None:
    route = CapabilityRouter().route("前回の記憶を確認してください")

    assert route.mode == RoutingMode.SCOPED
    assert route.capabilities == frozenset({Capability.MEMORY_READ})


def test_router_keeps_ambiguous_requests_open_for_llm_choice() -> None:
    route = CapabilityRouter().route("どうすればよいですか")

    assert route.mode == RoutingMode.OPEN


def test_router_does_not_treat_vague_test_as_process_request() -> None:
    route = CapabilityRouter().route("テスト")

    assert route.mode == RoutingMode.OPEN
    assert route.capabilities == frozenset()


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
