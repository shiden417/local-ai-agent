from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class Capability(str, Enum):
    WORKSPACE_READ = "workspace_read"
    WORKSPACE_WRITE = "workspace_write"
    PROCESS = "process"
    MEMORY_READ = "memory_read"
    MEMORY_WRITE = "memory_write"


class RoutingMode(str, Enum):
    DIRECT = "direct"
    SCOPED = "scoped"
    OPEN = "open"


@dataclass(frozen=True)
class CapabilityRoute:
    mode: RoutingMode
    capabilities: frozenset[Capability]


class CapabilityRouter:
    """Choose a tool scope, not a concrete tool action.

    Direct mode is intentionally conservative and only covers obvious
    conversational inputs. Scoped mode recognizes strong operational intent
    and exposes the matching capability families. Ambiguous requests use Open
    mode so the LLM still has access to the complete registered capability
    set and can decide whether a tool is actually necessary.
    """

    _DIRECT_PATTERNS: tuple[str, ...] = (
        r"^\s*(こんにちは|こんばんは|おはよう|やあ)[！!。\s]*$",
        r"^\s*(ありがとう|どうもありがとう|thanks|thank you)[！!。\s]*$",
        r"^\s*(さようなら|またね|bye)[！!。\s]*$",
        r"^\s*(元気ですか|元気？|元気\?)\s*$",
    )

    _PATTERNS: dict[Capability, tuple[str, ...]] = {
        Capability.WORKSPACE_READ: (
            r"フォルダ.{0,12}(中|一覧|何がある|調べ|確認)",
            r"ディレクトリ.{0,12}(中|一覧|何がある|調べ|確認)",
            r"ファイル.{0,12}(内容|中身|読ん|開い|読み取|確認|調べ)",
            r"(README|AGENTS\.md|pyproject\.toml|requirements\.txt|\.csproj|\.slnx?)",
            r"\b(workspace|repository|repo)\b.{0,20}(inspect|list|read|check|contents)",
            r"\b(folder|directory)\b.{0,20}(list|inspect|contents|files)",
            r"\b(file)\b.{0,20}(read|open|inspect|contents)",
        ),
        Capability.WORKSPACE_WRITE: (
            r"(作成|作って|つくって|生成|作り|新規).{0,12}(ファイル|コード|README|HTML|JSON|設定)?",
            r"(編集|変更|修正|書き換え|書換え|追加|削除|作成|作って|つくって|生成|新規).{0,8}(ファイル|コード|README|HTML|JSON|設定)?",
            r"(ファイル|コード|README|HTML|JSON|設定).{0,8}(編集|変更|修正|書き換え|追加|削除|作成|生成|新規)",
            r"\b(edit|modify|fix|change|update|write|create|delete|remove)\b",
        ),
        Capability.PROCESS: (
            r"(テスト|pytest).{0,12}(実行|走らせ|回し|して)",
            r"(ビルド|build).{0,12}(実行|して)?",
            r"(コマンド|PowerShell).{0,12}(実行|打|走らせ|して)",
            r"(起動|停止|インストール).{0,12}(して|する|を)",
            r"\b(pytest|powershell|git)\b",
            r"\b(run|execute|build|install|command)\b.{0,12}\b(it|this|test|project|command)?",
        ),
        Capability.MEMORY_READ: (
            r"(以前|前回|過去|覚えている|記憶|メモリ).{0,15}(確認|調べ|教え|思い出|検索)?",
            r"\b(previous|past|remember|memory)\b",
        ),
        Capability.MEMORY_WRITE: (
            r"(覚えておいて|覚えていて|記録して|保存して|今後も).{0,15}",
            r"\b(remember this|save this|for future)\b",
        ),
    }

    def route(self, task_text: str) -> CapabilityRoute:
        text = task_text.strip()
        if not text:
            return CapabilityRoute(
                mode=RoutingMode.DIRECT,
                capabilities=frozenset(),
            )

        if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in self._DIRECT_PATTERNS):
            return CapabilityRoute(
                mode=RoutingMode.DIRECT,
                capabilities=frozenset(),
            )

        capabilities = {
            capability
            for capability, patterns in self._PATTERNS.items()
            if any(
                re.search(pattern, text, flags=re.IGNORECASE)
                for pattern in patterns
            )
        }

        # Local file edits normally require inspection first. Keep the
        # capability scope explicit so the LLM can read the target before
        # choosing the mutating tool.
        if Capability.WORKSPACE_WRITE in capabilities:
            capabilities.add(Capability.WORKSPACE_READ)

        capabilities = frozenset(capabilities)

        if capabilities:
            return CapabilityRoute(
                mode=RoutingMode.SCOPED,
                capabilities=capabilities,
            )

        return CapabilityRoute(
            mode=RoutingMode.OPEN,
            capabilities=frozenset(),
        )

    def detect(self, task_text: str) -> set[Capability]:
        """Compatibility helper returning the scoped capability set."""
        return set(self.route(task_text).capabilities)
