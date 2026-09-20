import json


def parse_tool_call(text: str) -> dict | None:
    """Parse the agent's JSON tool-call protocol."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None

    if not isinstance(data, dict) or data.get("name") != "execute_command":
        return None

    arguments = data.get("arguments")
    if not isinstance(arguments, dict):
        return None

    command = arguments.get("command")
    if not isinstance(command, str) or not command.strip():
        return None

    return {"name": "execute_command", "command": command}
