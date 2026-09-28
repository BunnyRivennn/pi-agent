"""工具输出的截断工具函数。

上下文窗口是稀缺资源：一次 `grep` 可能匹配到几万行，一次 `ls` 可能列出几万个
文件。所有工具的输出都必须有上限，并且在截断时**明确告诉模型被截断了、怎么拿
剩下的**——沉默截断会让模型基于残缺信息做判断。
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "DEFAULT_MAX_BYTES",
    "GREP_MAX_LINE_LENGTH",
    "TruncationResult",
    "truncate_head",
    "truncate_line",
    "format_size",
]

#: 单次工具输出的字节上限。
DEFAULT_MAX_BYTES = 50_000
#: grep 单行的字符上限；超长行通常是压缩后的 JS/最小化资源，全量返回毫无价值。
GREP_MAX_LINE_LENGTH = 400


@dataclass(frozen=True, slots=True)
class TruncationResult:
    """截断结果。"""

    content: str
    truncated: bool
    original_bytes: int


def format_size(num_bytes: int) -> str:
    """把字节数格式化成便于阅读的字符串。"""
    if num_bytes >= 1024 * 1024:
        return f"{num_bytes / (1024 * 1024):.1f}MB"
    if num_bytes >= 1024:
        return f"{num_bytes // 1024}KB"
    return f"{num_bytes}B"


def truncate_head(
    text: str,
    *,
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_lines: int | None = None,
) -> TruncationResult:
    """保留开头，截掉超出限额的部分。

    与 ``bash`` 的中间截断不同：搜索/列举类结果前面的条目最相关，
    尾部通常是噪音，所以保头不保尾。
    """
    encoded = text.encode("utf-8", errors="replace")
    original_bytes = len(encoded)
    truncated = False

    if max_lines is not None:
        lines = text.split("\n")
        if len(lines) > max_lines:
            text = "\n".join(lines[:max_lines])
            truncated = True
            encoded = text.encode("utf-8", errors="replace")

    if len(encoded) > max_bytes:
        # 按字节切后可能切坏多字节字符，用 ignore 丢掉残片。
        text = encoded[:max_bytes].decode("utf-8", errors="ignore")
        truncated = True

    return TruncationResult(content=text, truncated=truncated, original_bytes=original_bytes)


def truncate_line(line: str, max_length: int = GREP_MAX_LINE_LENGTH) -> tuple[str, bool]:
    """截断过长的单行，返回 ``(文本, 是否被截断)``。"""
    if len(line) <= max_length:
        return line, False
    return line[:max_length] + "…", True


def build_notice(notices: list[str]) -> str:
    """把截断提示拼成统一的尾注格式。"""
    return f"\n\n[{'. '.join(notices)}]" if notices else ""
