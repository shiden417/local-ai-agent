from pathlib import Path

from tools.create_file import create_file
from tools.delete_file import delete_file
from tools.edit_file import edit_file


SUPPORTED_OPERATIONS = ("create", "edit", "delete")


def file_mutation(
    working_directory: str | Path,
    arguments: dict,
) -> dict:
    """Apply one explicit mutation to a local file."""
    operation = str(arguments.get("operation", "")).strip().lower()

    if operation == "create":
        return create_file(working_directory, arguments)

    if operation == "edit":
        return edit_file(working_directory, arguments)

    if operation == "delete":
        return delete_file(working_directory, arguments)

    return {
        "ok": False,
        "error": (
            f"Unsupported file mutation operation: {operation or '<missing>'}. "
            f"Supported operations: {', '.join(SUPPORTED_OPERATIONS)}"
        ),
    }
