from pathlib import Path


def resolve_workspace_path(
    working_directory: str | Path,
    relative_path: str,
) -> Path:
    """Resolve a path and ensure it stays inside the Agent workspace."""
    root = Path(working_directory).resolve()

    if not relative_path.strip():
        raise ValueError("path must not be empty")

    requested = Path(relative_path)
    if requested.is_absolute():
        raise ValueError("absolute paths are not allowed")

    resolved = (root / requested).resolve()

    if not resolved.is_relative_to(root):
        raise ValueError("path escapes the Agent workspace")

    return resolved


def to_workspace_relative(working_directory: str | Path, path: Path) -> str:
    root = Path(working_directory).resolve()
    return path.resolve().relative_to(root).as_posix()
