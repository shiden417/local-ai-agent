from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class RequestMode(str, Enum):
    DIRECT = "direct"
    TASK = "task"


@dataclass(frozen=True)
class RequestClassification:
    mode: RequestMode


class RequestClassifier:
    """Identify only whether input is ordinary conversation or an Agent task.

    Tool selection itself is intentionally delegated to the model's Tool
    Calling. The classifier does not infer capabilities or concrete tools.
    """

    _DIRECT_PATTERNS: tuple[str, ...] = (
        r"^\s*(こんにちは|こんばんは|おはよう|やあ)[！!。\s]*$",
        r"^\s*(ありがとう|どうもありがとう|thanks|thank you)[！!。\s]*$",
        r"^\s*(さようなら|またね|bye)[！!。\s]*$",
        r"^\s*(元気ですか|元気？|元気\?)\s*$",
    )

    _TASK_PATTERNS: tuple[str, ...] = (
        r"(作成|作って|生成|新規|編集|変更|修正|削除|書き換え)",
        r"(実行|走らせ|動かして|起動|停止|インストール|ビルド|テストして)",
        r"(調査|調べて|検索|探して|確認して|一覧を|見てください|変換して|解析して|抽出して|加工して|処理して)",
        r"(Web|web|ネット|インターネット).{0,20}(検索|調べ|探)",
        r"(最新|現在|今日|明日|価格|ニュース|天気).{0,20}(情報|教えて|調べ|検索|確認)?",
        r"(ファイル|フォルダ|ディレクトリ|プロジェクト|workspace|repository|repo).{0,20}(確認|調査|一覧|中身|内容|見て)",
        r"(pytest|powershell|git|python)\b.{0,20}(実行|run|execute|test|build|install)",
        r"(覚えておいて|記録して|保存して)",
    )

    def classify(self, request_text: str) -> RequestClassification:
        text = request_text.strip()
        if not text:
            return RequestClassification(RequestMode.DIRECT)

        if any(
            re.search(pattern, text, flags=re.IGNORECASE)
            for pattern in self._DIRECT_PATTERNS
        ):
            return RequestClassification(RequestMode.DIRECT)

        if any(
            re.search(pattern, text, flags=re.IGNORECASE)
            for pattern in self._TASK_PATTERNS
        ):
            return RequestClassification(RequestMode.TASK)

        return RequestClassification(RequestMode.DIRECT)
