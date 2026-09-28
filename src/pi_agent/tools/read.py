"""``read`` 工具：读取文件内容。

相比官方 ``read.ts`` 的两处有意偏离：

1. **编码探测**。官方硬编码 UTF-8；我们这边大量存在 GBK/GB2312 的中文遗留文件，
   死读 UTF-8 会得到乱码或直接抛异常。这里按 utf-8-sig → utf-8 → gb18030 →
   latin-1 顺序试，并在输出里标注实际使用的编码，让模型知道自己读到的是什么。
   gb18030 是 GBK/GB2312 的超集，一次覆盖。
2. **目录支持**。路径是目录时列出条目，省掉一次 ``bash ls`` 往返。
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from ..agent_core.types import AgentTool, AgentToolResult, TextContent
from .paths import resolve_in_root
from .registry import register_tool

__all__ = [
    "create_read_tool",
    "decode_bytes",
    "READ_PROMPT_SNIPPET",
    "READ_PROMPT_GUIDELINES",
]

READ_PROMPT_SNIPPET = "Read file contents (a directory path lists its entries)"
READ_PROMPT_GUIDELINES = (
    "Use read to examine files instead of shelling out to cat/sed/head.",
    "Read a file before editing it — edit requires the exact current text.",
)

#: 单次返回的上限，防止一个大文件撑爆上下文。
DEFAULT_MAX_LINES = 2000
DEFAULT_MAX_BYTES = 50_000

#: 编码探测顺序；gb18030 是 GBK/GB2312 的超集，latin-1 永不失败因此兜底。
_ENCODINGS = ("utf-8-sig", "utf-8", "gb18030", "latin-1")

_SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "Path to the file to read (relative or absolute)",
        },
        "offset": {
            "type": "integer",
            "description": "Line number to start reading from (1-indexed)",
        },
        "limit": {
            "type": "integer",
            "description": "Maximum number of lines to read",
        },
    },
    "required": ["path"],
}


def decode_bytes(raw: bytes) -> tuple[str, str]:
    """把字节解码成文本，返回 ``(文本, 实际编码)``。

    按 :data:`_ENCODINGS` 顺序试探。``grep`` 也复用它，好让中文 GBK 文件
    能被搜到——ripgrep 这类工具会把它们当二进制直接跳过。
    """
    for encoding in _ENCODINGS:
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    # latin-1 不会走到这里，保险起见留个替换式兜底。
    return raw.decode("utf-8", errors="replace"), "utf-8 (with replacements)"


def _list_directory(path: Path, limit: int) -> str:
    """列目录条目，目录带 ``/`` 后缀。"""
    try:
        entries = sorted(
            path.iterdir(),
            key=lambda p: (not p.is_dir(), p.name.lower()),
        )
    except PermissionError as exc:
        raise RuntimeError(f"Permission denied listing directory: {path}") from exc

    shown = entries[:limit]
    lines = [f"{e.name}/" if e.is_dir() else e.name for e in shown]
    if len(entries) > len(shown):
        lines.append(f"... (+{len(entries) - len(shown)} more entries)")
    if not lines:
        lines.append("(empty directory)")
    return "\n".join(lines)


def _read_text_file(
    path: Path,
    offset: int | None,
    limit: int | None,
    max_lines: int,
    max_bytes: int,
) -> str:
    raw = path.read_bytes()
    truncated_bytes = False
    if len(raw) > max_bytes:
        raw = raw[:max_bytes]
        truncated_bytes = True

    text, encoding = decode_bytes(raw)
    lines = text.splitlines()
    total = len(lines)

    start = max((offset or 1) - 1, 0)
    count = limit if limit is not None else max_lines
    count = max(min(count, max_lines), 0)
    window = lines[start : start + count]

    header = f"({encoding})" if not encoding.startswith("utf-8") else ""
    body = "\n".join(window)

    notes: list[str] = []
    if truncated_bytes:
        notes.append(f"truncated at {max_bytes} bytes")
    shown_end = start + len(window)
    if shown_end < total:
        notes.append(f"{total - shown_end} more lines below (use offset={shown_end + 1})")
    if start > 0:
        notes.append(f"started at line {start + 1}")

    parts = [p for p in (header, body) if p]
    result = "\n".join(parts) if header else body
    if notes:
        result = f"{result}\n\n[{'; '.join(notes)}]"
    return result


@register_tool("read")
def create_read_tool(
    cwd: Path,
    *,
    root: Path | None = None,
    allow_outside_root: bool = False,
    max_lines: int = DEFAULT_MAX_LINES,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> AgentTool:
    """构造 ``read`` 工具。

    Args:
        cwd: 相对路径的基准目录。
        root: 允许访问的根目录；``None`` 表示不限制。
        allow_outside_root: 放行根目录之外的路径。
        max_lines: 单次返回的最大行数。
        max_bytes: 单次读取的最大字节数。
    """

    async def execute(
        tool_call_id: str,
        params: Mapping[str, Any],
        abort_event: asyncio.Event | None = None,
        on_update: Callable[[AgentToolResult[Any]], None] | None = None,
    ) -> AgentToolResult[Any]:
        del tool_call_id, on_update
        raw_path = str(params["path"])
        offset = params.get("offset")
        limit = params.get("limit")

        target = resolve_in_root(raw_path, cwd, root, allow_outside_root=allow_outside_root)

        if abort_event is not None and abort_event.is_set():
            raise RuntimeError("Operation aborted")

        def _work() -> tuple[str, dict[str, Any]]:
            if target.is_dir():
                listing = _list_directory(target, limit or max_lines)
                return listing, {"path": str(target), "kind": "directory"}
            if not target.exists():
                raise RuntimeError(f"File not found: {target}")
            text = _read_text_file(
                target,
                int(offset) if offset is not None else None,
                int(limit) if limit is not None else None,
                max_lines,
                max_bytes,
            )
            return text, {"path": str(target), "kind": "file"}

        # 文件 IO 是阻塞的，丢到线程里避免卡住事件循环。
        text, details = await asyncio.to_thread(_work)

        return AgentToolResult(content=[TextContent(text=text)], details=details)

    return AgentTool(
        name="read",
        label="read",
        description=(
            "Read the contents of a file. Output is truncated to a maximum number of lines "
            "and bytes; use offset/limit to page through large files. If the path is a "
            "directory, its entries are listed instead."
        ),
        execute=execute,
        parameters=_SCHEMA,
        prompt_snippet=READ_PROMPT_SNIPPET,
        prompt_guidelines=READ_PROMPT_GUIDELINES,
    )
