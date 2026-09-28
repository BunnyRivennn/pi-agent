"""``grep`` 工具：在文件内容中搜索模式。

官方走 ripgrep 二进制；这里用纯 Python 实现，不要求用户额外装东西。
代价是大仓库上更慢，收益是零外部依赖、且能复用我们自己的编码探测逻辑
（ripgrep 遇到 GBK 文件会当成二进制跳过）。
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from ..agent_core.types import AgentTool, AgentToolResult, TextContent
from .ignore import load_ignore_rules, walk_files
from .paths import resolve_in_root
from .read import decode_bytes
from .registry import register_tool
from .truncate import (
    DEFAULT_MAX_BYTES,
    GREP_MAX_LINE_LENGTH,
    build_notice,
    format_size,
    truncate_head,
    truncate_line,
)

__all__ = ["create_grep_tool", "GREP_PROMPT_SNIPPET"]

GREP_PROMPT_SNIPPET = "Search file contents for patterns (respects .gitignore)"

DEFAULT_LIMIT = 100
#: 超过这个大小的文件不搜：多半是数据/二进制，搜了也没意义。
MAX_FILE_BYTES = 10 * 1024 * 1024

_SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "properties": {
        "pattern": {"type": "string", "description": "Search pattern (regex or literal)"},
        "path": {
            "type": "string",
            "description": "Directory or file to search (default: current directory)",
        },
        "glob": {
            "type": "string",
            "description": "Only search files matching this glob, e.g. '*.py'",
        },
        "ignoreCase": {"type": "boolean", "description": "Case-insensitive search"},
        "literal": {
            "type": "boolean",
            "description": "Treat pattern as a literal string instead of a regex",
        },
        "context": {
            "type": "integer",
            "description": "Lines of context to show before and after each match",
        },
        "limit": {
            "type": "integer",
            "description": f"Maximum number of matches (default: {DEFAULT_LIMIT})",
        },
    },
    "required": ["pattern"],
}


def _looks_binary(data: bytes) -> bool:
    """靠 NUL 字节判定二进制——和 grep/ripgrep 的启发式一致。"""
    return b"\x00" in data[:8000]


@register_tool("grep")
def create_grep_tool(
    cwd: Path,
    *,
    root: Path | None = None,
    allow_outside_root: bool = False,
    use_gitignore: bool = True,
) -> AgentTool:
    """构造 ``grep`` 工具。"""

    async def execute(
        tool_call_id: str,
        params: Mapping[str, Any],
        abort_event: asyncio.Event | None = None,
        on_update: Callable[[AgentToolResult[Any]], None] | None = None,
    ) -> AgentToolResult[Any]:
        del tool_call_id, on_update
        pattern = str(params["pattern"])
        raw_path = str(params.get("path") or ".")
        glob = params.get("glob")
        ignore_case = bool(params.get("ignoreCase"))
        literal = bool(params.get("literal"))
        context = max(0, int(params.get("context") or 0))
        limit = max(1, int(params.get("limit") or DEFAULT_LIMIT))

        target = resolve_in_root(raw_path, cwd, root, allow_outside_root=allow_outside_root)

        flags = re.IGNORECASE if ignore_case else 0
        try:
            regex = re.compile(re.escape(pattern) if literal else pattern, flags)
        except re.error as exc:
            raise RuntimeError(
                f"Invalid regex {pattern!r}: {exc}. Use literal=true to search for it verbatim."
            ) from exc

        def _work() -> tuple[str, dict[str, Any]]:
            if not target.exists():
                raise RuntimeError(f"Path not found: {target}")

            if target.is_dir():
                rules = load_ignore_rules(target, use_gitignore=use_gitignore)
                candidates = walk_files(target, rules)
                search_root = target
            else:
                candidates = iter([(target, target.name)])
                search_root = target.parent

            from .find import matches_glob

            rows: list[str] = []
            match_count = 0
            limit_reached = False
            lines_truncated = False
            files_with_matches = 0

            for absolute, relative in candidates:
                if abort_event is not None and abort_event.is_set():
                    raise RuntimeError("Operation aborted")
                if limit_reached:
                    break
                if glob and not matches_glob(relative, str(glob)):
                    continue

                try:
                    if absolute.stat().st_size > MAX_FILE_BYTES:
                        continue
                    data = absolute.read_bytes()
                except (OSError, ValueError):
                    continue
                if _looks_binary(data):
                    continue

                try:
                    text, _encoding = decode_bytes(data)
                except (UnicodeDecodeError, LookupError):
                    continue

                lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
                file_hit = False
                for index, line in enumerate(lines):
                    if not regex.search(line):
                        continue
                    if match_count >= limit:
                        limit_reached = True
                        break
                    match_count += 1
                    file_hit = True

                    line_number = index + 1
                    if context == 0:
                        shown, was_cut = truncate_line(line, GREP_MAX_LINE_LENGTH)
                        lines_truncated = lines_truncated or was_cut
                        rows.append(f"{relative}:{line_number}: {shown}")
                        continue

                    start = max(0, index - context)
                    end = min(len(lines), index + context + 1)
                    for cursor in range(start, end):
                        shown, was_cut = truncate_line(lines[cursor], GREP_MAX_LINE_LENGTH)
                        lines_truncated = lines_truncated or was_cut
                        # 匹配行用 ':' 分隔，上下文行用 '-'，沿用 GNU grep 惯例。
                        sep = ":" if cursor == index else "-"
                        rows.append(f"{relative}{sep}{cursor + 1}{sep} {shown}")
                    rows.append("--")

                if file_hit:
                    files_with_matches += 1

            if match_count == 0:
                return (
                    "No matches found",
                    {"pattern": pattern, "path": str(search_root), "count": 0},
                )

            truncation = truncate_head("\n".join(rows), max_bytes=DEFAULT_MAX_BYTES)
            details: dict[str, Any] = {
                "pattern": pattern,
                "path": str(search_root),
                "count": match_count,
                "files": files_with_matches,
            }

            notices: list[str] = []
            if limit_reached:
                notices.append(
                    f"{limit} matches limit reached. Use limit={limit * 2} for more, "
                    "or refine the pattern"
                )
                details["match_limit_reached"] = limit
            if truncation.truncated:
                notices.append(f"{format_size(DEFAULT_MAX_BYTES)} limit reached")
                details["truncated"] = True
            if lines_truncated:
                notices.append(
                    f"Some lines truncated to {GREP_MAX_LINE_LENGTH} chars. "
                    "Use the read tool to see full lines"
                )
                details["lines_truncated"] = True

            return truncation.content + build_notice(notices), details

        text, details = await asyncio.to_thread(_work)
        return AgentToolResult(content=[TextContent(text=text)], details=details)

    return AgentTool(
        name="grep",
        label="grep",
        description=(
            "Search file contents for a pattern. Returns matching lines with file paths and "
            "line numbers. Respects .gitignore and skips binary files. Truncated to "
            f"{DEFAULT_LIMIT} matches or {DEFAULT_MAX_BYTES // 1024}KB, whichever is hit first. "
            f"Long lines are truncated to {GREP_MAX_LINE_LENGTH} chars."
        ),
        execute=execute,
        parameters=_SCHEMA,
        prompt_snippet=GREP_PROMPT_SNIPPET,
    )
