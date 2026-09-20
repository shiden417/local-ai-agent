from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class Capability(str, Enum):
    WORKSPACE_READ = "workspace_read"
    WORKSPACE_WRITE = "workspace_write"
    PROCESS = "process"
    SCRIPT_EXECUTION = "script_execution"
    CAPABILITY_MANAGEMENT = "capability_management"
    MEMORY_READ = "memory_read"
    MEMORY_WRITE = "memory_write"
    WEB_SEARCH = "web_search"
    AGENT_CONTROL = "agent_control"


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
    mode so the Runtime can answer conversationally without exposing operational
    Tools speculatively.
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
            r"設定.{0,12}(確認|調べ|内容)",
            r"ディレクトリ.{0,12}(中|一覧|何がある|調べ|確認)",
            r"ファイル.{0,12}(内容|中身|読ん|開い|読み取|確認|調べ)",
            r"(README|AGENTS\.md|pyproject\.toml|requirements\.txt|\.csproj|\.slnx?)",
            r"\b(workspace|repository|repo)\b.{0,20}(inspect|list|read|check|contents)",
            r"\b(folder|directory)\b.{0,20}(list|inspect|contents|files)",
            r"\b(file)\b.{0,20}(read|open|inspect|contents)",
        ),
        Capability.WORKSPACE_WRITE: (
            r"(作成|作って|つくって|生成|作り|新規).{0,12}(ファイル|コード|README|HTML|JSON|設定|ドキュメント)",
            r"(ファイル|コード|README|HTML|JSON|設定|ドキュメント).{0,12}(編集|変更|修正|書き換え|書換え|追加|削除|作成|生成|新規|更新)",
            r"(編集|変更|修正|書き換え|書換え|追加|削除|作成|生成|新規|更新).{0,12}(ファイル|コード|README|HTML|JSON|設定|ドキュメント)",
            r"[A-Za-z0-9_.\\/-]+\\.[A-Za-z0-9_-]{1,12}.{0,12}(編集|変更|修正|書き換え|書換え|追加|削除|作成|生成|更新)(?:して|してください|したい|する)?",
            r"(バグ|不具合|問題|issue|bug).{0,20}(修正|直して|直す|fix|resolve)",
            r"(修正|直して|直す|fix|resolve).{0,20}(バグ|不具合|問題|issue|bug)",
            r"\b(edit|modify|fix|change|update|write|create|delete|remove)\b.{0,20}\b(file|code|readme|config|document|bug|issue)\b",
        ),
        Capability.SCRIPT_EXECUTION: (
            r"(Python|python|スクリプト|script).{0,20}(実行|動か|コード|スクリプト|実行して|動かして)",
            r"(変換|解析|パース|抽出|加工|処理).{0,20}(して|する|したい|してください|お願い)",
            r"^\s*(調査|調べ物|リサーチ|research)\s*$",
            r"\b(convert|parse|extract|transform|process|generate)\b",
        ),
        Capability.CAPABILITY_MANAGEMENT: (
            r"(Tool|ツール|プラグイン|plugin|capability|能力).{0,20}(追加|作成|生成|改善|修正|登録|導入|有効|増や)",
            r"(できるように|できるようにな).{0,20}(追加|Tool|ツール|能力|プラグイン)",
            r"\b(add|create|generate|improve|install|enable)\b.{0,20}\b(tool|plugin|capability)\b",
        ),
        Capability.PROCESS: (
            r"(作業|タスク).{0,12}(完了|終了|進め|実行|して|してください)",
            r"^\s*(調査|作業|アクション)\s*$",
            r"(調査|作業|アクション).{0,12}(実行|確認|調べ|進め|して|してください|お願い)",
            r"(終わらない|続いている|途中の).{0,4}作業",
            r"(テスト|pytest).{0,12}(実行|走らせ|回し|して)",
            r"(ビルド|build).{0,12}(実行|して)?",
            r"(コマンド|PowerShell).{0,12}(実行|打|走らせ|して)",
            r"(起動|停止|インストール).{0,12}(して|する|を)",
            r"\b(pytest|powershell|git)\b",
            r"\b(run|execute|build|install|command)\b.{0,12}\b(it|this|test|project|command)?",
        ),
        Capability.WEB_SEARCH: (
            r"(Web|web|WEB).{0,20}(検索|search|調べ|探して|情報)",
            r"(ネット|インターネット|ネット上).{0,20}(検索|調べ|探して|情報)",
            r"(検索|調べ|探して).{0,20}(Web|web|ネット|インターネット|最新|ニュース|公式サイト)",
            r".{1,80}(について|に関して).{0,20}(調べ|調査|リサーチ|検索)",
            r"(最新|現在|今日|最近).{0,20}(情報|ニュース).{0,20}(検索|調べ|探して)?",
            r"(今日|現在|今|本日|明日|あす|明後日).{0,20}(天気|気温|降水|雨|雪|台風|警報|気象)",
            r"(天気|気温|降水|雨|雪|台風|警報|気象).{0,20}(今日|現在|今|本日|明日|あす|明後日|最新)",
            r"(今日|現在|最新).{0,20}(価格|値段|料金|株価|為替|レート|営業時間|運行|交通|イベント|スポーツ|試合|販売状況)",
            r"\b(search|web search|internet|online)\b",
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

        # Explicit requests to add or change the Agent's capabilities
        # are routed to capability management only. Do not expose unrelated
        # workspace/script tools at the same time.
        if Capability.CAPABILITY_MANAGEMENT in capabilities:
            return CapabilityRoute(
                mode=RoutingMode.SCOPED,
                capabilities=frozenset({Capability.CAPABILITY_MANAGEMENT}),
            )

        # Operational investigations may need both process-oriented
        # actions and workspace inspection. Keep simple action requests narrow.
        if (
            Capability.PROCESS in capabilities
            and re.search(r"(調査|問題点|現在の状態|原因|確認)", text)
        ):
            capabilities.add(Capability.WORKSPACE_READ)

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
