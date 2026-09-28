"""``edit`` 工具的文本匹配与 diff 生成。

对齐官方 ``edit-diff.ts`` 的核心约束：

1. 所有 ``old_text`` 都对**原始内容**匹配，不是增量应用——否则模型得在脑子里
   模拟前几个 edit 的结果才能写对后面的。
2. 每个 ``old_text`` 必须**唯一**；出现多次直接报错，而不是猜第一个。
3. 全部匹配定位后**按位置排序检查重叠**，重叠报错。
4. 统一按 LF 匹配，写回时恢复原文件的行尾风格。

报错信息刻意写得详细：它会作为 tool result 回到模型眼前，是模型自我纠正的
唯一依据。含糊的报错会让模型反复瞎试。
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass

__all__ = [
    "Edit",
    "AppliedEdits",
    "apply_edits",
    "detect_line_ending",
    "normalize_to_lf",
    "restore_line_endings",
    "split_bom",
    "generate_unified_patch",
]

_BOM = "\ufeff"


@dataclass(frozen=True, slots=True)
class Edit:
    """一次精确替换。"""

    old_text: str
    new_text: str


@dataclass(frozen=True, slots=True)
class AppliedEdits:
    """替换结果（均为 LF 规范化后的文本）。"""

    base_content: str
    new_content: str


def split_bom(text: str) -> tuple[str, str]:
    """剥离 BOM，返回 ``(bom, 正文)``。

    模型不会在 ``old_text`` 里带上不可见的 BOM，所以匹配前必须剥掉，
    写回时再原样加回去。
    """
    if text.startswith(_BOM):
        return _BOM, text[len(_BOM) :]
    return "", text


def detect_line_ending(text: str) -> str:
    """探测主导的行尾风格，返回 ``"\\r\\n"`` 或 ``"\\n"``。

    只要出现过 CRLF 就认为是 CRLF 文件——混合行尾的文件按多数派处理会
    在写回时把另一半改掉，制造大片无意义 diff。
    """
    return "\r\n" if "\r\n" in text else "\n"


def normalize_to_lf(text: str) -> str:
    """把 CRLF / CR 统一成 LF。"""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def restore_line_endings(text: str, ending: str) -> str:
    """把 LF 文本还原成指定行尾风格。"""
    if ending == "\n":
        return text
    return text.replace("\n", ending)


def _count_occurrences(haystack: str, needle: str) -> int:
    """数不重叠的出现次数。"""
    if not needle:
        return 0
    return haystack.count(needle)


def _describe(index: int, total: int) -> str:
    """单个 edit 时不提下标，避免报错信息噪音。"""
    return "The edit" if total == 1 else f"edits[{index}]"


@dataclass(frozen=True, slots=True)
class _Matched:
    edit_index: int
    start: int
    length: int
    new_text: str


def apply_edits(normalized_content: str, edits: list[Edit], path: str) -> AppliedEdits:
    """对 LF 规范化后的内容应用全部替换。

    Raises:
        ValueError: 任一 edit 为空、匹配不到、匹配多次、彼此重叠，或整体无变化。
    """
    if not edits:
        raise ValueError("Edit tool input is invalid. edits must contain at least one replacement.")

    normalized = [
        Edit(old_text=normalize_to_lf(e.old_text), new_text=normalize_to_lf(e.new_text))
        for e in edits
    ]
    total = len(normalized)

    for i, edit in enumerate(normalized):
        if not edit.old_text:
            raise ValueError(
                f"{_describe(i, total)} has an empty oldText in {path}. "
                "oldText must be a non-empty exact excerpt of the file."
            )

    matched: list[_Matched] = []
    for i, edit in enumerate(normalized):
        occurrences = _count_occurrences(normalized_content, edit.old_text)
        if occurrences == 0:
            raise ValueError(
                f"{_describe(i, total)} did not match anything in {path}. "
                "oldText must match the file exactly, including whitespace and indentation. "
                "Read the file again to get the current content."
            )
        if occurrences > 1:
            raise ValueError(
                f"{_describe(i, total)} matched {occurrences} places in {path}, but it must be "
                "unique. Extend oldText with surrounding lines until it identifies exactly one "
                "location."
            )
        start = normalized_content.index(edit.old_text)
        matched.append(
            _Matched(
                edit_index=i,
                start=start,
                length=len(edit.old_text),
                new_text=edit.new_text,
            )
        )

    # 按命中位置排序后检查相邻区间是否交叠。
    matched.sort(key=lambda m: m.start)
    for previous, current in zip(matched, matched[1:], strict=False):
        if previous.start + previous.length > current.start:
            raise ValueError(
                f"edits[{previous.edit_index}] and edits[{current.edit_index}] overlap in {path}. "
                "Merge them into one edit or target disjoint regions."
            )

    # 从后往前replace，这样前面的下标不会被已应用的替换挪动。
    new_content = normalized_content
    for m in reversed(matched):
        new_content = new_content[: m.start] + m.new_text + new_content[m.start + m.length :]

    if new_content == normalized_content:
        raise ValueError(
            f"The {total} edit(s) produced no change in {path}. "
            "newText is identical to oldText — drop the edit or supply the intended new content."
        )

    return AppliedEdits(base_content=normalized_content, new_content=new_content)


def generate_unified_patch(
    path: str,
    old_content: str,
    new_content: str,
    context_lines: int = 4,
) -> str:
    """生成标准 unified diff。"""
    diff = difflib.unified_diff(
        old_content.splitlines(keepends=True),
        new_content.splitlines(keepends=True),
        fromfile=path,
        tofile=path,
        n=context_lines,
    )
    return "".join(diff)


def first_changed_line(old_content: str, new_content: str) -> int | None:
    """第一处变化在新文件里的行号（1 起）；无变化返回 ``None``。

    供编辑器/UI 跳转定位用。
    """
    old_lines = old_content.splitlines()
    new_lines = new_content.splitlines()
    matcher = difflib.SequenceMatcher(a=old_lines, b=new_lines, autojunk=False)
    for tag, _i1, _i2, j1, _j2 in matcher.get_opcodes():
        if tag != "equal":
            return j1 + 1
    return None
