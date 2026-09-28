"""``find`` 工具：按 glob 模式查找文件。"""

from __future__ import annotations

import asyncio
import fnmatch
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from ..agent_core.types import AgentTool, AgentToolResult, TextContent
from .ignore import load_ignore_rules, walk_files
from .paths import resolve_in_root
from .registry import register_tool
from .truncate import DEFAULT_MAX_BYTES, build_notice, format_size, truncate_head

__all__ = ["create_find_tool", "FIND_PROMPT_SNIPPET"]

FIND_PROMPT_SNIPPET = "Find files by glob pattern (respects .gitignore)"

DEFAULT_LIMIT = 1000

_SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "properties": {
        "pattern": {
            "type": "string",
            "description": "Glob pattern, e.g. '*.py', '**/*.json', or 'src/**/*_test.py'",
        },
        "path": {
            "type": "string",
            "description": "Directory to search in (default: current directory)",
        },
        "limit": {
            "type": "integer",
            "description": f"Maximum number of results (default: {DEFAULT_LIMIT})",
        },
    },
    "required": ["pattern"],
}


def matches_glob(relative_path: str, pattern: str) -> bool:
    """判断相对路径是否匹配 glob 模式。

    模型写 ``*.py`` 时通常想要「所有层级的 .py」，而不是严格的「仅根目录下」。
    所以不含 ``/`` 的模式对**文件名**匹配，含 ``/`` 的才对整个相对路径匹配。
    """
    normalized = relative_path.replace("\\", "/")
    if "/" not in pattern:
        return fnmatch.fnmatch(normalized.rsplit("/", 1)[-1], pattern)

    if fnmatch.fnmatch(normalized, pattern):
        return True
    # 让 'src/**/*.py' 也能匹配 'src/a.py'（** 匹配零层目录）。
    collapsed = pattern.replace("/**/", "/")
    return collapsed != pattern and fnmatch.fnmatch(normalized, collapsed)


@register_tool("find")
def create_find_tool(
    cwd: Path,
    *,
    root: Path | None = None,
    allow_outside_root: bool = False,
    use_gitignore: bool = True,
) -> AgentTool:
    """构造 ``find`` 工具。"""

    async def execute(
        tool_call_id: str,
        params: Mapping[str, Any],
        abort_event: asyncio.Event | None = None,
        on_update: Callable[[AgentToolResult[Any]], None] | None = None,
    ) -> AgentToolResult[Any]:
        del tool_call_id, on_update
        pattern = str(params["pattern"])
        raw_path = str(params.get("path") or ".")
        limit = max(1, int(params.get("limit") or DEFAULT_LIMIT))

        search_root = resolve_in_root(raw_path, cwd, root, allow_outside_root=allow_outside_root)

        def _work() -> tuple[str, dict[str, Any]]:
            if not search_root.exists():
                raise RuntimeError(f"Path not found: {search_root}")
            if not search_root.is_dir():
                raise RuntimeError(f"Not a directory: {search_root}")

            rules = load_ignore_rules(search_root, use_gitignore=use_gitignore)
            results: list[str] = []
            limit_reached = False

            for _absolute, relative in walk_files(search_root, rules):
                # 中断检查放在循环里：大目录树遍历可能持续数秒。
                if abort_event is not None and abort_event.is_set():
                    raise RuntimeError("Operation aborted")
                if not matches_glob(relative, pattern):
                    continue
                if len(results) >= limit:
                    limit_reached = True
                    break
                results.append(relative)

            if not results:
                return (
                    f"No files matching {pattern!r}",
                    {"pattern": pattern, "path": str(search_root), "count": 0},
                )

            truncation = truncate_head("\n".join(results), max_bytes=DEFAULT_MAX_BYTES)
            details: dict[str, Any] = {
                "pattern": pattern,
                "path": str(search_root),
                "count": len(results),
            }

            notices: list[str] = []
            if limit_reached:
                notices.append(f"{limit} results limit reached. Use limit={limit * 2} for more")
                details["result_limit_reached"] = limit
            if truncation.truncated:
                notices.append(f"{format_size(DEFAULT_MAX_BYTES)} limit reached")
                details["truncated"] = True

            return truncation.content + build_notice(notices), details

        text, details = await asyncio.to_thread(_work)
        return AgentToolResult(content=[TextContent(text=text)], details=details)

    return AgentTool(
        name="find",
        label="find",
        description=(
            "Search for files by glob pattern. Returns paths relative to the search directory. "
            "Respects .gitignore and skips heavy directories like node_modules. "
            f"Truncated to {DEFAULT_LIMIT} results or {DEFAULT_MAX_BYTES // 1024}KB, "
            "whichever is hit first."
        ),
        execute=execute,
        parameters=_SCHEMA,
        prompt_snippet=FIND_PROMPT_SNIPPET,
    )
