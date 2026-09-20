from __future__ import annotations

import re
from enum import Enum


class Capability(str, Enum):
    WORKSPACE_READ = "workspace_read"
    WORKSPACE_WRITE = "workspace_write"
    PROCESS = "process"
    MEMORY_READ = "memory_read"
    MEMORY_WRITE = "memory_write"


class CapabilityRouter:
    """Lightweight pre-filter for capability scope, not tool selection.

    The router only decides which capability families are relevant enough to
    expose. The LLM still decides whether to call a specific tool and which
    tool to call. The implementation is intentionally replaceable by a
    semantic classifier later without changing the ToolRegistry or Runtime.
    """

    _PATTERNS: dict[Capability, tuple[str, ...]] = {
        Capability.WORKSPACE_READ: (
            r"フォルダ.{0,12}(中|一覧|何がある|調べ|確認)",
            r"ディレクトリ.{0,12}(中|一覧|何がある|調べ|確認)",
            r"ファイル.{0,12}(内容|中身|読ん|開い|読み取|確認|調べ)",
            r"(README|AGENTS\.md|pyproject\.toml|requirements\.txt|\.csproj|\.slnx?)",
            r"(workspace|repository|repo).{0,20}(inspect|list|read|check|contents)",
            r"(folder|directory).{0,20}(list|inspect|contents|files)",
            r"(file).{0,20}(read|open|inspect|contents)",
        ),
        Capability.WORKSPACE_WRITE: (
            r"(編集|変更|修正|書き換え|書換え|追加|削除).{0,8}(ファイル|コード|README|設定)?",
            r"(ファイル|コード|README|設定).{0,8}(編集|変更|修正|書き換え|追加|削除)",
            r"(edit|modify|fix|change|update|write|create|delete|remove)",
        ),
        Capability.PROCESS: (
            r"(実行|コマンド|テスト|ビルド|起動|停止|インストール).{0,15}",
            r"(run|execute|test|build|install|command|powershell)",
            r"git",
        ),
        Capability.MEMORY_READ: (
            r"(以前|前回|過去|覚えている|記憶|メモリ).{0,15}(確認|調べ|教え|思い出|検索)?",
            r"(previous|past|remember|memory)",
        ),
        Capability.MEMORY_WRITE: (
            r"(覚えておいて|覚えていて|記録して|保存して|今後も).{0,15}",
            r"(remember this|save this|for future)",
        ),
    }

    def detect(self, task_text: str) -> set[Capability]:
        text = task_text.casefold()
        return {
            capability
            for capability, patterns in self._PATTERNS.items()
            if any(re.search(pattern, text) for pattern in patterns)
        }
