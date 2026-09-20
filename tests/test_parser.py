from agent.parser import parse_tool_call


def test_parse_execute_command() -> None:
    assert parse_tool_call(
        '{"name":"execute_command","arguments":{"command":"Get-Date"}}'
    ) == {"name": "execute_command", "command": "Get-Date"}


def test_parse_invalid_json() -> None:
    assert parse_tool_call("not json") is None


def test_parse_other_tool() -> None:
    assert parse_tool_call(
        '{"name":"read_file","arguments":{"path":"README.md"}}'
    ) is None
