from agent.task_requirements import classify_task_requirements


def test_read_only_task_can_explicitly_require_process_execution() -> None:
    requirements = classify_task_requirements(
        "ファイルを変更せず、execute_command Toolを使って "
        'python -m pytest -q を実行し、終了コード0を確認してください。'
    )

    assert requirements.read_only is True
    assert requirements.process_execution is True
    assert requirements.required_process_tool == "execute_command"


def test_plain_read_only_investigation_still_forbids_process_execution() -> None:
    requirements = classify_task_requirements(
        "config.json の値だけ確認してください。ファイルは変更せず、"
        "無関係なファイルの内容を読み込まないでください。"
    )

    assert requirements.read_only is True
    assert requirements.process_execution is False


def test_required_symbols_track_explicit_function_additions() -> None:
    requirements = classify_task_requirements(
        "calculator に subtract(a, b) を追加し、tests/test_calculator.py も変更して、"
        "python -m pytest -q を実行してください。"
    )

    assert requirements.required_symbols == ("subtract",)



def test_latest_function_requirement_overrides_obsolete_removal() -> None:
    requirements = classify_task_requirements(
        "calculator に subtract(a, b) を追加してください。"
        "tests/test_calculator.py にもテストを追加してください。"
        "要件が更新されました。先ほどの subtract は不要です。"
        "subtract を削除し、代わりに divide(a, b) を追加してください。"
    )

    assert requirements.required_symbols == ("divide",)
