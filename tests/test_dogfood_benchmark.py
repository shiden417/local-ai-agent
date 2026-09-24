from pathlib import Path

from tools.dogfood_benchmark import (
    _add_required_regression_test,
    _inject_truncation_bug,
)


def test_inject_truncation_bug_changes_only_expected_source(tmp_path: Path) -> None:
    observation = tmp_path / "agent" / "observation.py"
    observation.parent.mkdir()
    original = (
        "def truncate_text(text, max_chars=8000):\n"
        "    return text[:head] + marker + text[-tail:], True\n"
    )
    observation.write_text(original, encoding="utf-8")

    before, _ = _inject_truncation_bug(tmp_path)

    assert before == original
    assert observation.read_text(encoding="utf-8") == (
        "def truncate_text(text, max_chars=8000):\n"
        "    return text[:head] + marker + text[-head:], True\n"
    )


def test_add_required_regression_test_is_deterministic(tmp_path: Path) -> None:
    tests = tmp_path / "tests" / "test_observation.py"
    tests.parent.mkdir()
    original = "from agent.observation import truncate_text\n"
    tests.write_text(original, encoding="utf-8")

    result = _add_required_regression_test(tmp_path)

    content = tests.read_text(encoding="utf-8")
    assert result == content
    assert content.startswith(original)
    assert "test_truncate_text_respects_max_chars_for_small_limits" in content
