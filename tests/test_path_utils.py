from pathlib import Path

import pytest

from tools.path_utils import resolve_workspace_path, to_display_path


def test_relative_path_stays_inside_workspace(tmp_path: Path) -> None:
    assert resolve_workspace_path(tmp_path, "sub/file.txt") == (
        tmp_path / "sub/file.txt"
    ).resolve()


def test_relative_parent_path_cannot_escape_workspace(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="escapes"):
        resolve_workspace_path(tmp_path, "../outside.txt")


def test_explicit_absolute_path_is_supported(tmp_path: Path) -> None:
    external = tmp_path.parent / "external"
    assert resolve_workspace_path(tmp_path, str(external)) == external.resolve()


def test_display_path_is_relative_inside_workspace(tmp_path: Path) -> None:
    target = tmp_path / "file.txt"
    assert to_display_path(tmp_path, target) == "file.txt"


def test_display_path_is_absolute_outside_workspace(tmp_path: Path) -> None:
    target = tmp_path.parent / "external" / "file.txt"
    assert to_display_path(tmp_path, target) == str(target.resolve())
