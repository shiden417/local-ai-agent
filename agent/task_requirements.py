from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class TaskRequirements:
    """Deterministic requirements and mutation constraints inferred from the task wording."""

    read_only: bool
    file_mutation: bool
    process_execution: bool
    test_verification: bool
    mutation_forbidden: bool = False
    protected_paths: tuple[str, ...] = ()
    required_process_tool: str | None = None


_FILE_CONTEXT_RE = re.compile(
    r"(?:ファイル|file|\.py(?![A-Za-z0-9_])|\.txt(?![A-Za-z0-9_])|"
    r"\.json(?![A-Za-z0-9_])|\.md(?![A-Za-z0-9_])|\.yaml(?![A-Za-z0-9_])|"
    r"\.yml(?![A-Za-z0-9_])|\.csv(?![A-Za-z0-9_])|workspace|path|directory|コード)",
    re.IGNORECASE,
)

_JAPANESE_MUTATION_RE = re.compile(
    r"(?:追加|追記|作成|修正|変更|編集|削除|書き換え|保存|書き込み|リネーム|名前変更)"
    r"\s*(?:してください|して|し、|した|しろ|する|します|を)"
)

_JAPANESE_MUTATION_NEGATION_RE = re.compile(
    r"(?:ファイル(?:は|を)?\s*)?"
    r"(?:追加|作成|修正|変更|編集|削除|書き換え|保存|書き込み)"
    r"\s*(?:せず(?:に)?|することなく|しない(?:で(?:ください|下さい)?)?|しません|禁止)"
)

_JAPANESE_IMPLEMENT_RE = re.compile(
    r"実装\s*(?:してください|して|し、|した|しろ|する|します|を)"
)

_ENGLISH_MUTATION_RE = re.compile(
    r"\b(?:please\s+)?(?:add|create|modify|change|edit|delete|update|write)"
    r"\s+(?:a|an|the|new|this|that|file|folder|directory|line|code|test)\b",
    re.IGNORECASE,
)

_ENGLISH_IMPLEMENT_RE = re.compile(
    r"\b(?:please\s+)?implement"
    r"\s+(?:a|an|the|new|this|that|feature|function|method|class)\b",
    re.IGNORECASE,
)

_READ_INTENT_RE = re.compile(
    r"(調査|調べ|検索|探して|確認|閲覧|読み|分析|"
    r"diagnos|investigat|inspect|search|review|read|check|verify)",
    re.IGNORECASE,
)

_NO_CHANGE_RE = re.compile(
    r"(変更しない|変更なし|変更は不要|変更禁止|変更せず|変更せずに|変更することなく|"
    r"改変しない|改変禁止|改変せず|"
    r"修正しない|修正禁止|修正せず|編集しない|編集禁止|編集せず|ファイルを変更しない|"
    r"do not\s+(?:modify|change|edit)|don't\s+(?:modify|change|edit)|"
    r"without\s+(?:modifying|changing|editing)|read[- ]?only|no changes?)",
    re.IGNORECASE,
)

_PROCESS_INTENT_RE = re.compile(
    r"(?:コマンド(?:を|の)?実行|コマンド実行|"
    r"プロセス(?:を|の)?実行|実行(?:してください|して|し、|する|します)|"
    r"\bexecute\b|\brun\b|command execution|process execution)",
    re.IGNORECASE,
)

_EXPLICIT_PROCESS_CONTEXT_RE = re.compile(
    r"(?:コマンド|プロセス|PowerShell|terminal|shell|execute_command|"
    r"command|process)",
    re.IGNORECASE,
)

_PYTHON_COMMAND_RE = re.compile(
    r"\bpython(?:\.exe)?\s+(?:-[a-z]+\b|[^\s]+\.py\b)",
    re.IGNORECASE,
)

_TEST_REQUEST_RE = re.compile(
    r"(?:回帰|pytest|regression|全テスト"
    r"|テスト(?:を|の|が)?\s*(?:実行|実施|走らせ|成功|失敗|確認|検証)"
    r"|\btest(?:ing|s)?\s+(?:suite|case|coverage|run|result)\b)",
    re.IGNORECASE,
)

_FILE_PATH_TOKEN = (
    r"(?:[A-Za-z]:[\\/])?(?:[A-Za-z0-9_.-]+[\\/])*"
    r"[A-Za-z0-9_.-]+\.(?:py|txt|json|md|yaml|yml|csv)"
)

_PROTECTED_PATH_RE = re.compile(
    rf"(?P<path>{_FILE_PATH_TOKEN})\s*(?:は|を|が)?\s*"
    r"(?:絶対に\s*)?(?:変更|修正|編集|削除|書き換え|更新|上書き)"
    r"\s*(?:しない|しません|禁止|不要|しないで(?:ください|下さい)?)"
    rf"|(?P<english_path>{_FILE_PATH_TOKEN})\s+"
    r"(?:must\s+not|should\s+not|do\s+not|don't)\s+"
    r"(?:modify|change|edit|delete|update|overwrite)",
    re.IGNORECASE,
)

_GLOBAL_NO_FILE_MUTATION_RE = re.compile(
    r"(?:"
    r"ファイル(?:は|を)?(?:絶対に)?(?:変更|修正|編集|削除|書き換え|更新)\s*"
    r"(?:しない|しません|禁止|しないで(?:ください|下さい)?)"
    r"|"
    r"(?:ワークスペース|workspace)(?:内|の)?(?:ファイル|files?)\s*"
    r"(?:は|を)?\s*(?:絶対に)?(?:変更|修正|編集|削除|書き換え|更新|modify|change|edit|delete)\s*"
    r"(?:しない|しません|禁止|しないで(?:ください|下さい)?|not|never)"
    r"|"
    r"(?:do\s+not|don't|without|never)\s+(?:modify|change|edit|delete|update)\s+"
    r"(?:any\s+)?(?:workspace\s+)?files?"
    r")",
    re.IGNORECASE,
)


def _has_positive_mutation_intent(text: str) -> bool:
    """Detect positive mutation requests without treating negative constraints as actions."""
    positive_text = _JAPANESE_MUTATION_NEGATION_RE.sub("", text)
    positive_text = re.sub(
        r"\b(?:do not|don't)\s+(?:modify|change|edit)\b",
        "",
        positive_text,
        flags=re.IGNORECASE,
    )
    return bool(
        _JAPANESE_MUTATION_RE.search(positive_text)
        or _JAPANESE_IMPLEMENT_RE.search(positive_text)
        or _ENGLISH_MUTATION_RE.search(positive_text)
        or _ENGLISH_IMPLEMENT_RE.search(positive_text)
    )


def classify_task_requirements(goal: str) -> TaskRequirements:
    raw_text = str(goal)
    text = raw_text.casefold()
    file_context = bool(_FILE_CONTEXT_RE.search(text))
    positive_mutation = _has_positive_mutation_intent(text)

    read_only = bool(
        _READ_INTENT_RE.search(text)
        and _NO_CHANGE_RE.search(text)
        and not positive_mutation
    )

    file_mutation = bool(file_context and positive_mutation)

    protected_paths: list[str] = []
    for match in _PROTECTED_PATH_RE.finditer(raw_text):
        path = match.group("path") or match.group("english_path")
        if path and path not in protected_paths:
            protected_paths.append(path)

    mutation_forbidden = bool(_GLOBAL_NO_FILE_MUTATION_RE.search(raw_text))
    if mutation_forbidden:
        # A global no-mutation constraint overrides incidental save/write wording,
        # including Memory-related language that should not imply file mutation.
        file_mutation = False

    process_context = bool(
        _EXPLICIT_PROCESS_CONTEXT_RE.search(text)
        or _PYTHON_COMMAND_RE.search(text)
    )
    process_execution = bool(
        _PROCESS_INTENT_RE.search(text) and process_context
    )

    test_verification = bool(_TEST_REQUEST_RE.search(text))
    explicit_command_context = bool(
        re.search(
            r"(execute_command|コマンド|powershell|terminal|shell|"
            r"python(?:\.exe)?\s+-[a-z]+|pytest|dotnet|npm|git)",
            text,
            re.IGNORECASE,
        )
    )
    required_process_tool = (
        "execute_command"
        if "execute_command" in text or (process_execution and explicit_command_context)
        else None
    )

    return TaskRequirements(
        read_only=read_only,
        file_mutation=file_mutation,
        process_execution=process_execution,
        test_verification=test_verification,
        mutation_forbidden=mutation_forbidden,
        protected_paths=tuple(protected_paths),
        required_process_tool=required_process_tool,
    )
