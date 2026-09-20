from pathlib import Path


def resolve_workspace_path(
    working_directory: str | Path,
    requested_path: str,
) -> Path:
    """Resolve a local path, allowing explicit absolute paths.

    Relative paths must remain inside the Agent workspace. Absolute paths are
    allowed so that a user can explicitly target another local directory; the
    Runtime is responsible for confirmation of such operations.
    """
    root = Path(working_directory).resolve()

    if not requested_path.strip():
        raise ValueError("path must not be empty")

    requested = Path(requested_path)
    if requested.is_absolute():
        return requested.resolve()

    resolved = (root / requested).resolve()

    if not resolved.is_relative_to(root):
        raise ValueError("path escapes the Agent workspace")

    return resolved


def to_display_path(working_directory: str | Path, path: Path) -> str:
    """Return a workspace-relative path or an absolute path for external files."""
    root = Path(working_directory).resolve()
    resolved = path.resolve()
    if resolved.is_relative_to(root):
        return resolved.relative_to(root).as_posix() or "."
    return str(resolved)


def to_workspace_relative(working_directory: str | Path, path: Path) -> str:
    """Backward-compatible workspace-relative formatter."""
    root = Path(working_directory).resolve()
    return path.resolve().relative_to(root).as_posix()
