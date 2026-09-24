from pathlib import Path

from tools.python_symbol_edit import python_symbol_edit


def test_add_function_appends_one_top_level_function(tmp_path: Path) -> None:
    target = tmp_path / "calculator.py"
    target.write_text(
        "def add(a, b):\n"
        "    return a + b\n",
        encoding="utf-8",
    )

    result = python_symbol_edit(
        tmp_path,
        {
            "operation": "add_function",
            "path": "calculator.py",
            "symbol": "square",
            "function_code": "def square(a):\n    return a ** 2",
        },
    )

    assert result["ok"] is True
    assert target.read_text(encoding="utf-8").count("def square(") == 1
    compile(target.read_text(encoding="utf-8"), str(target), "exec")


def test_add_function_rejects_duplicate_symbol(tmp_path: Path) -> None:
    target = tmp_path / "calculator.py"
    target.write_text(
        "def square(a):\n    return a ** 2\n",
        encoding="utf-8",
    )

    result = python_symbol_edit(
        tmp_path,
        {
            "operation": "add_function",
            "path": "calculator.py",
            "symbol": "square",
            "function_code": "def square(a):\n    return a ** 2",
        },
    )

    assert result["ok"] is False
    assert "already exists" in result["error"]


def test_remove_function_removes_only_requested_top_level_function(tmp_path: Path) -> None:
    target = tmp_path / "calculator.py"
    target.write_text(
        "def add(a, b):\n"
        "    return a + b\n"
        "\n"
        "def subtract(a, b):\n"
        "    return a - b\n"
        "\n"
        "def multiply(a, b):\n"
        "    return a * b\n",
        encoding="utf-8",
    )

    result = python_symbol_edit(
        tmp_path,
        {
            "operation": "remove_function",
            "path": "calculator.py",
            "symbol": "subtract",
        },
    )

    content = target.read_text(encoding="utf-8")
    assert result["ok"] is True
    assert "def subtract(" not in content
    assert "def add(" in content
    assert "def multiply(" in content


def test_rename_identifier_changes_python_names_not_strings_or_comments(tmp_path: Path) -> None:
    target = tmp_path / "calculator.py"
    target.write_text(
        "# multiply should remain in this comment\n"
        'label = "multiply"\n'
        "def multiply(a, b):\n"
        "    return multiply_helper(a, b)\n"
        "\n"
        "def multiply_helper(a, b):\n"
        "    return a * b\n"
        "\n"
        "result = multiply(2, 3)\n",
        encoding="utf-8",
    )

    result = python_symbol_edit(
        tmp_path,
        {
            "operation": "rename_identifier",
            "path": "calculator.py",
            "symbol": "multiply",
            "new_name": "product",
        },
    )

    content = target.read_text(encoding="utf-8")
    assert result["ok"] is True
    assert "def product(a, b):" in content
    assert "result = product(2, 3)" in content
    assert "# multiply should remain in this comment" in content
    assert 'label = "multiply"' in content
    assert "def multiply_helper(" in content


def test_ensure_from_import_updates_existing_import(tmp_path: Path) -> None:
    target = tmp_path / "test_calculator.py"
    target.write_text(
        "from src.calculator import add, multiply\n"
        "\n"
        "def test_square():\n"
        "    assert square(2) == 4\n",
        encoding="utf-8",
    )

    result = python_symbol_edit(
        tmp_path,
        {
            "operation": "ensure_from_import",
            "path": "test_calculator.py",
            "symbol": "square",
            "module": "src.calculator",
        },
    )

    content = target.read_text(encoding="utf-8")
    assert result["ok"] is True
    assert "from src.calculator import add, multiply, square" in content


def test_ensure_from_import_inserts_after_future_import(tmp_path: Path) -> None:
    target = tmp_path / "test_calculator.py"
    target.write_text(
        '"""tests"""\n'
        "from __future__ import annotations\n"
        "\n"
        "def test_square():\n"
        "    assert square(2) == 4\n",
        encoding="utf-8",
    )

    result = python_symbol_edit(
        tmp_path,
        {
            "operation": "ensure_from_import",
            "path": "test_calculator.py",
            "symbol": "square",
            "module": "src.calculator",
        },
    )

    lines = target.read_text(encoding="utf-8").splitlines()
    assert result["ok"] is True
    assert lines[:3] == [
        '"""tests"""',
        "from __future__ import annotations",
        "from src.calculator import square",
    ]
