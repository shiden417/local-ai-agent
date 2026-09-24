from agent.runtime import AgentRuntime


def test_follow_up_uses_latest_read_only_intent_over_inherited_mutation() -> None:
    requirements = AgentRuntime._effective_task_requirements(
        "そのファイルを確認し、2行目が Session Context works になっていることだけ確認してください。",
        "session.txt に2行目として『Session Context works』を追加してください。"
        "既存の1行目は変更しないでください。"
        "Follow-up request: そのファイルを確認し、2行目が Session Context works になっていることだけ確認してください。",
        is_follow_up=True,
    )

    assert requirements.file_mutation is False
    assert requirements.process_execution is False
    assert requirements.required_mutation_paths == ("session.txt",)


def test_follow_up_keeps_targets_but_uses_latest_function_requirement() -> None:
    requirements = AgentRuntime._effective_task_requirements(
        "要件が更新されました。先ほどの subtract は最終要件では不要です。"
        "subtract を削除し、代わりに divide(a, b) を追加してください。"
        "tests/test_calculator.py も最終要件に合わせて更新して、"
        "python -m pytest -q を実行してください。",
        "calculator に subtract(a, b) を追加してください。"
        "src/calculator.py と tests/test_calculator.py を必要に応じて変更し、"
        "python -m pytest -q を実行して全テスト成功を確認してください。"
        "Follow-up request: 要件が更新されました。先ほどの subtract は最終要件では不要です。"
        "subtract を削除し、代わりに divide(a, b) を追加してください。"
        "tests/test_calculator.py も最終要件に合わせて更新して、"
        "python -m pytest -q を実行してください。",
        is_follow_up=True,
    )

    assert requirements.file_mutation is True
    assert requirements.required_symbols == ("divide",)
    assert requirements.required_mutation_paths == (
        "src/calculator.py",
        "tests/test_calculator.py",
    )
    assert requirements.required_process_tool == "execute_command"
