from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

from agent.capability_router import Capability
from agent.observation import truncate_text
from agent.tool_registry import ToolDefinition, ToolRegistry


DEFAULT_PLUGIN_ROOT = Path.home() / ".local-ai-agent" / "plugins"
DEFAULT_PLUGIN_TIMEOUT_SECONDS = 15
MAX_PLUGIN_TIMEOUT_SECONDS = 30
MAX_PLUGIN_SOURCE_CHARS = 20_000
MAX_PLUGIN_OUTPUT_CHARS = 8_000


class PluginValidationError(ValueError):
    """Raised when a plugin package is invalid."""


class PluginManager:
    """Stage, validate, promote, and load local process-isolated plugins."""

    def __init__(self, root: str | Path = DEFAULT_PLUGIN_ROOT) -> None:
        self.root = Path(root).expanduser().resolve()
        self.quarantine_root = self.root / "quarantine"
        self.enabled_root = self.root / "enabled"
        self.quarantine_root.mkdir(parents=True, exist_ok=True)
        self.enabled_root.mkdir(parents=True, exist_ok=True)

    def stage(
        self,
        plugin_id: str,
        manifest: dict[str, Any],
        source: str,
    ) -> dict[str, Any]:
        plugin_id = _validate_plugin_id(plugin_id)
        self._validate_manifest(manifest, plugin_id)
        source = str(source)
        if not source.strip():
            raise PluginValidationError("plugin source must not be empty")
        if len(source) > MAX_PLUGIN_SOURCE_CHARS:
            raise PluginValidationError(
                f"plugin source exceeds {MAX_PLUGIN_SOURCE_CHARS} characters"
            )

        plugin_dir = self.quarantine_root / plugin_id
        if plugin_dir.exists():
            raise PluginValidationError(
                f"quarantine plugin already exists: {plugin_id}"
            )

        self._validate_source(source)
        plugin_dir.mkdir(parents=True)
        try:
            (plugin_dir / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (plugin_dir / "plugin.py").write_text(
                source,
                encoding="utf-8",
            )
        except OSError:
            shutil.rmtree(plugin_dir, ignore_errors=True)
            raise

        return {
            "ok": True,
            "status": "quarantined",
            "plugin_id": plugin_id,
            "path": str(plugin_dir),
        }

    def test_candidate(
        self,
        plugin_id: str,
        manifest: dict[str, Any],
        source: str,
        test_arguments: dict[str, Any],
        working_directory: str | Path,
    ) -> dict[str, Any]:
        plugin_id = _validate_plugin_id(plugin_id)
        try:
            self._validate_manifest(manifest, plugin_id)
            self._validate_source(source)
        except PluginValidationError as exc:
            return {"ok": False, "error": str(exc)}

        if not isinstance(test_arguments, dict):
            return {"ok": False, "error": "test_arguments must be an object"}

        required = manifest.get("parameters", {}).get("required", [])
        if not isinstance(required, list):
            return {
                "ok": False,
                "error": "plugin parameters required must be an array",
            }
        missing = [
            str(name)
            for name in required
            if name not in test_arguments
        ]
        if missing:
            return {
                "ok": False,
                "error": f"test_arguments missing required fields: {', '.join(missing)}",
            }

        import tempfile

        try:
            with tempfile.TemporaryDirectory(
                prefix=".agent-plugin-test-",
                dir=str(self.quarantine_root),
            ) as temporary_dir:
                plugin_file = Path(temporary_dir) / "plugin.py"
                plugin_file.write_text(source, encoding="utf-8")
                result = _execute_plugin(
                    plugin_file,
                    Path(working_directory).resolve(),
                    test_arguments,
                    int(
                        manifest.get(
                            "timeout_seconds",
                            DEFAULT_PLUGIN_TIMEOUT_SECONDS,
                        )
                    ),
                )
        except OSError as exc:
            return {
                "ok": False,
                "error": f"plugin candidate test failed to start: {exc}",
            }

        return {
            "ok": bool(result.get("ok")),
            "status": "tested",
            "plugin_id": plugin_id,
            "test_arguments": test_arguments,
            "result": result,
        }

    def promote(self, plugin_id: str) -> dict[str, Any]:
        plugin_id = _validate_plugin_id(plugin_id)
        source_dir = self.quarantine_root / plugin_id
        if not source_dir.is_dir():
            return {
                "ok": False,
                "error": f"quarantine plugin not found: {plugin_id}",
            }

        try:
            manifest = self._read_manifest(source_dir)
            source = self._read_source(source_dir)
            self._validate_manifest(manifest, plugin_id)
            self._validate_source(source)
        except (OSError, json.JSONDecodeError, PluginValidationError) as exc:
            return {"ok": False, "error": str(exc)}

        target_dir = self.enabled_root / plugin_id
        if target_dir.exists():
            return {
                "ok": False,
                "error": f"enabled plugin already exists: {plugin_id}",
            }

        try:
            shutil.move(str(source_dir), str(target_dir))
        except OSError as exc:
            return {
                "ok": False,
                "error": f"failed to promote plugin: {exc}",
            }

        return {
            "ok": True,
            "status": "enabled",
            "plugin_id": plugin_id,
            "path": str(target_dir),
            "tool_name": str(manifest["name"]),
        }

    def load_enabled(self, registry: ToolRegistry) -> list[dict[str, Any]]:
        loaded: list[dict[str, Any]] = []
        try:
            candidates = sorted(
                path for path in self.enabled_root.iterdir() if path.is_dir()
            )
        except OSError:
            return loaded

        for plugin_dir in candidates:
            try:
                tool = self.load_plugin(plugin_dir)
                if registry.get(tool.name) is not None:
                    continue
                registry.register(tool)
                loaded.append(
                    {
                        "plugin_id": plugin_dir.name,
                        "tool_name": tool.name,
                        "ok": True,
                    }
                )
            except (OSError, json.JSONDecodeError, PluginValidationError):
                continue

        return loaded

    def load_plugin(self, plugin_dir: str | Path) -> ToolDefinition:
        directory = Path(plugin_dir).resolve()
        manifest = self._read_manifest(directory)
        source = self._read_source(directory)
        plugin_id = directory.name

        self._validate_manifest(manifest, plugin_id)
        self._validate_source(source)

        try:
            capabilities = tuple(
                Capability(value)
                for value in manifest.get("capabilities", [])
            )
        except ValueError as exc:
            raise PluginValidationError(
                f"unsupported plugin capability: {exc}"
            ) from exc

        timeout = int(
            manifest.get(
                "timeout_seconds",
                DEFAULT_PLUGIN_TIMEOUT_SECONDS,
            )
        )

        return ToolDefinition(
            name=str(manifest["name"]),
            description=str(manifest["description"]),
            parameters=dict(manifest["parameters"]),
            handler=self._build_handler(directory, timeout),
            # Plugins always require confirmation in the current trust model.
            requires_confirmation=True,
            use_when=str(manifest.get("use_when", "")),
            avoid_when=str(manifest.get("avoid_when", "")),
            availability="on_demand",
            capabilities=capabilities,
            terminal_on_success=bool(manifest.get("terminal_on_success", False)),
        )

    def peek_manifest(self, plugin_id: str) -> dict[str, Any]:
        plugin_id = _validate_plugin_id(plugin_id)
        directory = self.quarantine_root / plugin_id
        return self._read_manifest(directory)

    def _build_handler(self, plugin_dir: Path, timeout: int):
        plugin_file = (plugin_dir / "plugin.py").resolve()

        def handler(working_directory: Path, arguments: dict[str, Any]) -> dict[str, Any]:
            return _execute_plugin(
                plugin_file,
                working_directory,
                arguments,
                timeout,
            )

        return handler

    @staticmethod
    def _validate_manifest(
        manifest: dict[str, Any],
        plugin_id: str,
    ) -> None:
        if not isinstance(manifest, dict):
            raise PluginValidationError("plugin manifest must be an object")

        required = ("name", "version", "description", "parameters")
        missing = [key for key in required if not str(manifest.get(key, "")).strip()]
        if missing:
            raise PluginValidationError(
                f"plugin manifest missing required fields: {', '.join(missing)}"
            )

        name = str(manifest["name"]).strip()
        if name != _validate_tool_name(name):
            raise PluginValidationError("plugin tool name is invalid")

        if manifest.get("entrypoint", "plugin.py") != "plugin.py":
            raise PluginValidationError(
                "plugin entrypoint must be plugin.py"
            )
        if manifest.get("function", "run") != "run":
            raise PluginValidationError(
                "plugin function must be run"
            )

        parameters = manifest["parameters"]
        if not isinstance(parameters, dict):
            raise PluginValidationError("plugin parameters must be an object")
        if parameters.get("type") != "object":
            raise PluginValidationError(
                "plugin parameters schema must have type=object"
            )

        capabilities = manifest.get("capabilities", [])
        if not isinstance(capabilities, list) or not capabilities or not all(
            isinstance(value, str) for value in capabilities
        ):
            raise PluginValidationError(
                "plugin capabilities must be a non-empty array of strings"
            )
        for value in capabilities:
            try:
                Capability(value)
            except ValueError as exc:
                raise PluginValidationError(
                    f"unsupported plugin capability: {value}"
                ) from exc

        timeout = manifest.get(
            "timeout_seconds",
            DEFAULT_PLUGIN_TIMEOUT_SECONDS,
        )
        if not isinstance(timeout, int):
            raise PluginValidationError("plugin timeout_seconds must be an integer")
        if not 1 <= timeout <= MAX_PLUGIN_TIMEOUT_SECONDS:
            raise PluginValidationError(
                f"plugin timeout_seconds must be between 1 and "
                f"{MAX_PLUGIN_TIMEOUT_SECONDS}"
            )

        if plugin_id != _validate_plugin_id(plugin_id):
            raise PluginValidationError("plugin id is invalid")

    @staticmethod
    def _validate_source(source: str) -> None:
        if not source.strip():
            raise PluginValidationError("plugin source must not be empty")

        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            raise PluginValidationError(
                f"plugin source has a syntax error: {exc}"
            ) from exc

        has_run = any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "run"
            for node in tree.body
        )
        if not has_run:
            raise PluginValidationError(
                "plugin source must define top-level run(arguments)"
            )

        blocked_modules = {
            "os",
            "pathlib",
            "shutil",
            "socket",
            "subprocess",
            "ctypes",
            "winreg",
            "requests",
            "httpx",
            "urllib",
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name.split(".")[0] for alias in node.names]
                if any(module in blocked_modules for module in modules):
                    raise PluginValidationError(
                        "plugin source imports a blocked module"
                    )
            elif isinstance(node, ast.ImportFrom):
                module = (node.module or "").split(".")[0]
                if module in blocked_modules:
                    raise PluginValidationError(
                        "plugin source imports a blocked module"
                    )
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in {
                    "eval",
                    "exec",
                    "compile",
                    "open",
                    "__import__",
                }:
                    raise PluginValidationError(
                        f"plugin source uses blocked builtin: {node.func.id}"
                    )

    @staticmethod
    def _read_manifest(plugin_dir: Path) -> dict[str, Any]:
        manifest_path = plugin_dir / "manifest.json"
        raw = manifest_path.read_text(encoding="utf-8")
        manifest = json.loads(raw)
        if not isinstance(manifest, dict):
            raise PluginValidationError("plugin manifest must be an object")
        return manifest

    @staticmethod
    def _read_source(plugin_dir: Path) -> str:
        return (plugin_dir / "plugin.py").read_text(encoding="utf-8")


def _execute_plugin(
    plugin_file: Path,
    working_directory: Path,
    arguments: dict[str, Any],
    timeout_seconds: int,
) -> dict[str, Any]:
    if not plugin_file.exists():
        return {"ok": False, "error": f"plugin file not found: {plugin_file}"}

    cwd = Path(working_directory).resolve()
    if not cwd.is_dir():
        return {"ok": False, "error": f"workspace not found: {cwd}"}

    payload = json.dumps(
        {"arguments": arguments},
        ensure_ascii=False,
    )

    environment = os.environ.copy()
    for key in (
        "PYTHONPATH",
        "PYTHONHOME",
        "PYTHONSTARTUP",
        "PYTHONUSERBASE",
    ):
        environment.pop(key, None)

    try:
        process = subprocess.Popen(
            [
                sys.executable,
                "-I",
                "-c",
                "import importlib.util\nimport json\nimport sys\n\nplugin_path = sys.argv[1]\nspec = importlib.util.spec_from_file_location(\"local_agent_plugin\", plugin_path)\nif spec is None or spec.loader is None:\n    raise RuntimeError(\"could not load plugin module\")\nmodule = importlib.util.module_from_spec(spec)\nspec.loader.exec_module(module)\nrun = getattr(module, \"run\", None)\nif not callable(run):\n    raise RuntimeError(\"plugin must define callable run(arguments)\")\npayload = json.loads(sys.stdin.read())\nresult = run(payload.get(\"arguments\", {}))\nif not isinstance(result, dict):\n    raise RuntimeError(\"plugin result must be a JSON object\")\nprint(json.dumps(result, ensure_ascii=False))\n",
                str(plugin_file),
            ],
            cwd=str(cwd),
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        try:
            stdout, stderr = process.communicate(
                input=payload,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            _terminate_process_tree(process.pid)
            stdout, stderr = process.communicate()
            stderr = (
                f"{stderr}\nPluginが{timeout_seconds}秒以内に終了しなかったため終了しました。"
            ).strip()
            return {
                "ok": False,
                "error": stderr,
                "timed_out": True,
            }

        stdout, stdout_truncated = truncate_text(
            stdout,
            MAX_PLUGIN_OUTPUT_CHARS,
        )
        stderr, _ = truncate_text(stderr, MAX_PLUGIN_OUTPUT_CHARS)

        if process.returncode != 0:
            return {
                "ok": False,
                "error": (
                    f"Plugin exited with code {process.returncode}: "
                    f"{stderr or 'no stderr'}"
                ),
            }

        try:
            result = json.loads(stdout)
        except json.JSONDecodeError as exc:
            return {
                "ok": False,
                "error": f"Plugin returned invalid JSON: {exc}",
                "stdout": stdout,
                "stderr": stderr,
            }

        if not isinstance(result, dict):
            return {
                "ok": False,
                "error": "Plugin result must be a JSON object",
            }

        if stdout_truncated:
            return {
                "ok": False,
                "error": "Plugin output exceeded the maximum size",
            }

        return result
    except OSError as exc:
        return {
            "ok": False,
            "error": f"Plugin execution failed: {exc}",
        }


def _terminate_process_tree(pid: int) -> None:
    if sys.platform.startswith("win"):
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return

    try:
        os.kill(pid, 9)
    except OSError:
        pass


def _validate_plugin_id(value: str) -> str:
    text = str(value).strip()
    if not text or len(text) > 64:
        raise PluginValidationError("plugin id must be 1-64 characters")
    if not all(character.isalnum() or character in {"-", "_"} for character in text):
        raise PluginValidationError(
            "plugin id may contain only letters, numbers, '-' and '_'"
        )
    return text


def _validate_tool_name(value: str) -> str:
    text = str(value).strip()
    if not text or len(text) > 64:
        raise PluginValidationError("plugin tool name must be 1-64 characters")
    if not all(
        character.isalnum() or character in {"-", "_"}
        for character in text
    ):
        raise PluginValidationError(
            "plugin tool name may contain only letters, numbers, '-' and '_'"
        )
    return text
