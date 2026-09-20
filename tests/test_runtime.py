from agent.runtime import AgentRuntime


def test_runtime_initializes_with_absolute_working_directory(tmp_path) -> None:
    runtime = AgentRuntime(tmp_path)
    assert runtime.working_directory.is_absolute()
    assert runtime.max_iterations == 10
    assert runtime.messages[0]["role"] == "system"
