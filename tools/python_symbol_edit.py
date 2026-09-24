from __future__ import annotations

import ast
import difflib
import io
import tokenize
from pathlib import Path
from typing import Any

from tools.path_utils import resolve_workspace_path, to_display_path
from tools.source_validation import validate_python_syntax


MAX_FILE_SIZE = 1_000_000
SUPPORTED_OPERATIONS = (
    "add_function",
    "remove_function",
    "rename_identifier",
    "ensure_from_import",
)


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="cp932", errors="replace")


def _validate_python_path(path: Path) -> str | None:
    if path.suffix.casefold() != ".py":
        return "python_symbol_edit only supports .py files."
    if not path.exists():
        return f"File does not exist: {path}"
    if not path.is_file():
        return f"Not a file: {path}"
    try:
        if path.stat().st_size > MAX_FILE_SIZE:
            return "File is too large to edit safely"
    except OSError as exc:
        return f"Unable to inspect file: {exc}"
    return None


def _parse_module(content: str) -> ast.Module | str:
    try:
        return ast.parse(content)
    except SyntaxError as exc:
        return f"Current file is not valid Python: {exc}"


def _function_nodes(
    tree: ast.Module,
    symbol: str,
) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    return [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == symbol
    ]


def _node_start_line(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    if node.decorator_list:
        return min(
            [node.lineno, *(decorator.lineno for decorator in node.decorator_list)]
        )
    return node.lineno


def _add_function(
    content: str,
    symbol: str,
    function_code: str,
) -> tuple[str, dict[str, object]]:
    tree_or_error = _parse_module(content)
    if isinstance(tree_or_error, str):
        return content, {"ok": False, "error": tree_or_error}
    if not symbol:
        return content, {"ok": False, "error": "symbol must not be empty"}
    if not function_code.strip():
        return content, {"ok": False, "error": "function_code must not be empty"}

    if _function_nodes(tree_or_error, symbol):
        return content, {
            "ok": False,
            "error": f"Top-level function already exists: {symbol}",
        }

    new_tree_or_error = _parse_module(function_code)
    if isinstance(new_tree_or_error, str):
        return content, {
            "ok": False,
            "error": f"function_code is not valid Python: {new_tree_or_error}",
        }

    top_level_functions = [
        node
        for node in new_tree_or_error.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    if len(new_tree_or_error.body) != 1 or len(top_level_functions) != 1:
        return content, {
            "ok": False,
            "error": "function_code must contain exactly one top-level function.",
        }

    function_node = top_level_functions[0]
    if function_node.name != symbol:
        return content, {
            "ok": False,
            "error": (
                f"function_code defines {function_node.name!r}, "
                f"but symbol is {symbol!r}."
            ),
        }

    suffix = "" if not content else ("\n" if content.endswith("\n") else "\n\n")
    new_content = content + suffix + function_code.strip() + "\n"
    return new_content, {
        "ok": True,
        "operation": "add_function",
        "symbol": symbol,
    }


def _remove_function(
    content: str,
    symbol: str,
) -> tuple[str, dict[str, object]]:
    tree_or_error = _parse_module(content)
    if isinstance(tree_or_error, str):
        return content, {"ok": False, "error": tree_or_error}

    matches = _function_nodes(tree_or_error, symbol)
    if not matches:
        return content, {
            "ok": False,
            "error": f"Top-level function not found: {symbol}",
        }
    if len(matches) > 1:
        return content, {
            "ok": False,
            "error": f"Top-level function appears multiple times: {symbol}",
        }

    node = matches[0]
    if node.end_lineno is None:
        return content, {
            "ok": False,
            "error": f"Unable to determine end of function: {symbol}",
        }

    lines = content.splitlines(keepends=True)
    start = _node_start_line(node) - 1
    end = node.end_lineno
    if end < len(lines) and not lines[end].strip():
        end += 1

    new_content = "".join(lines[:start] + lines[end:])
    if new_content and not new_content.endswith("\n"):
        new_content += "\n"

    return new_content, {
        "ok": True,
        "operation": "remove_function",
        "symbol": symbol,
    }


def _rename_identifier(
    content: str,
    symbol: str,
    new_name: str,
) -> tuple[str, dict[str, object]]:
    if not symbol or not new_name:
        return content, {"ok": False, "error": "symbol and new_name are required"}
    if not symbol.isidentifier() or not new_name.isidentifier():
        return content, {
            "ok": False,
            "error": "symbol and new_name must be valid Python identifiers",
        }
    if symbol == new_name:
        return content, {
            "ok": False,
            "error": "symbol and new_name must be different",
        }

    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(content).readline))
    except (tokenize.TokenError, IndentationError) as exc:
        return content, {
            "ok": False,
            "error": f"Unable to tokenize Python source: {exc}",
        }

    matches = [
        token
        for token in tokens
        if token.type == tokenize.NAME and token.string == symbol
    ]
    if not matches:
        return content, {
            "ok": False,
            "error": f"Identifier not found: {symbol}",
        }

    lines = content.splitlines(keepends=True)
    offsets: list[int] = []
    total = 0
    for line in lines:
        offsets.append(total)
        total += len(line)

    replacements: list[tuple[int, int]] = []
    for token in matches:
        start_line, start_col = token.start
        end_line, end_col = token.end
        if start_line != end_line:
            continue
        start = offsets[start_line - 1] + start_col
        end = offsets[end_line - 1] + end_col
        replacements.append((start, end))

    for start, end in reversed(replacements):
        content = content[:start] + new_name + content[end:]

    return content, {
        "ok": True,
        "operation": "rename_identifier",
        "symbol": symbol,
        "new_name": new_name,
        "replacements": len(replacements),
    }


def _ensure_from_import(
    content: str,
    module: str,
    symbol: str,
) -> tuple[str, dict[str, object]]:
    if not module or not symbol:
        return content, {
            "ok": False,
            "error": "module and symbol are required",
        }
    if not symbol.isidentifier():
        return content, {
            "ok": False,
            "error": "symbol must be a valid Python identifier",
        }

    tree_or_error = _parse_module(content)
    if isinstance(tree_or_error, str):
        return content, {"ok": False, "error": tree_or_error}

    for node in tree_or_error.body:
        if (
            isinstance(node, ast.ImportFrom)
            and node.level == 0
            and node.module == module
            and any(alias.name == symbol for alias in node.names)
        ):
            return content, {
                "ok": True,
                "operation": "ensure_from_import",
                "module": module,
                "symbol": symbol,
                "replacements": 0,
                "no_op": True,
            }

    for node in tree_or_error.body:
        if (
            isinstance(node, ast.ImportFrom)
            and node.level == 0
            and node.module == module
        ):
            if node.end_lineno is None:
                break

            lines = content.splitlines(keepends=True)
            line_index = node.lineno - 1
            current = lines[line_index].rstrip("\r\n")
            import_index = current.index("import")
            prefix = current[: import_index + len("import")]
            imported_text = current[import_index + len("import"):].strip()
            names = [
                item.strip()
                for item in imported_text.split(",")
                if item.strip()
            ]
            names.append(symbol)
            newline = "\r\n" if "\r\n" in lines[line_index] else "\n"
            lines[line_index] = f"{prefix} {', '.join(names)}{newline}"
            return "".join(lines), {
                "ok": True,
                "operation": "ensure_from_import",
                "module": module,
                "symbol": symbol,
                "replacements": 1,
            }

    lines = content.splitlines(keepends=True)
    insert_at = 0
    for node in tree_or_error.body:
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            insert_at = node.end_lineno or node.lineno
            continue
        if (
            isinstance(node, ast.ImportFrom)
            and node.level == 0
            and node.module == "__future__"
        ):
            insert_at = node.end_lineno or node.lineno
            continue
        break

    newline = "\r\n" if any("\r\n" in line for line in lines) else "\n"
    lines.insert(insert_at, f"from {module} import {symbol}{newline}")
    return "".join(lines), {
        "ok": True,
        "operation": "ensure_from_import",
        "module": module,
        "symbol": symbol,
        "replacements": 1,
    }


def python_symbol_edit(
    working_directory: str | Path,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """Apply a deterministic, Python-aware edit to one .py file."""
    operation = str(arguments.get("operation", "")).strip().lower()
    requested_path = str(arguments.get("path", "")).strip()

    if operation not in SUPPORTED_OPERATIONS:
        return {
            "ok": False,
            "error": (
                f"Unsupported operation: {operation or '<missing>'}. "
                f"Supported operations: {', '.join(SUPPORTED_OPERATIONS)}"
            ),
        }

    try:
        path = resolve_workspace_path(working_directory, requested_path)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}

    path_error = _validate_python_path(path)
    if path_error is not None:
        return {"ok": False, "error": path_error}

    try:
        content = _read_text(path)
    except OSError as exc:
        return {"ok": False, "error": f"Unable to read file: {exc}"}

    symbol = str(arguments.get("symbol", "")).strip()
    if operation == "add_function":
        new_content, result = _add_function(
            content,
            symbol,
            str(arguments.get("function_code", "")),
        )
    elif operation == "remove_function":
        new_content, result = _remove_function(content, symbol)
    elif operation == "rename_identifier":
        new_content, result = _rename_identifier(
            content,
            symbol,
            str(arguments.get("new_name", "")).strip(),
        )
    else:
        new_content, result = _ensure_from_import(
            content,
            str(arguments.get("module", "")).strip(),
            symbol,
        )

    if not result.get("ok"):
        return result

    if new_content == content:
        result["path"] = to_display_path(working_directory, path)
        return result

    validation_error = validate_python_syntax(path, new_content)
    if validation_error is not None:
        return {
            **result,
            "ok": False,
            "error": validation_error,
            "validation_failed": True,
            "path": to_display_path(working_directory, path),
        }

    try:
        path.write_text(new_content, encoding="utf-8", newline="")
    except OSError as exc:
        return {"ok": False, "error": f"Unable to write file: {exc}"}

    return {
        **result,
        "path": to_display_path(working_directory, path),
        "diff": "".join(
            difflib.unified_diff(
                content.splitlines(keepends=True),
                new_content.splitlines(keepends=True),
                fromfile=requested_path,
                tofile=requested_path,
            )
        ),
    }
