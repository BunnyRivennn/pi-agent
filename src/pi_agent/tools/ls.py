"""``ls`` 工具：列出目录内容。"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from ..agent_core.types import AgentTool, AgentToolResult, TextContent
from .paths import resolve_in_root
from .registry import register_tool
from .truncate import DEFAULT_MAX_BYTES, build_notice, format_size, truncate_head

__all__ = ["create_ls_tool", "LS_PROMPT_SNIPPET"]

LS_PROMPT_SNIPPET = "List directory contents"

DEFAULT_LIMIT = 500

_SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "Directory to list (default: current directory)",
        },
        "limit": {
            "type": "integer",
            "description": f"Maximum number of entries to return (default: {DEFAULT_LIMIT})",
        },
    },
}


@register_tool("ls")
def create_ls_tool(
    cwd: Path,
    *,
    root: Path | None = None,
    allow_outside_root: bool = False,
) -> AgentTool:
    """构造 ``ls`` 工具。"""

    async def execute(
        tool_call_id: str,
        params: Mapping[str, Any],
        abort_event: asyncio.Event | None = None,
        on_update: Callable[[AgentToolResult[Any]], None] | None = None,
    ) -> AgentToolResult[Any]:
        del tool_call_id, on_update
        raw_path = str(params.get("path") or ".")
        limit = int(params.get("limit") or DEFAULT_LIMIT)

        target = resolve_in_root(raw_path, cwd, root, allow_outside_root=allow_outside_root)

        if abort_event is not None and abort_event.is_set():
            raise RuntimeError("Operation aborted")

        def _work() -> tuple[str, dict[str, Any]]:
            if not target.exists():
                raise RuntimeError(f"Path not found: {target}")
            if not target.is_dir():
                raise RuntimeError(f"Not a directory: {target}")
            try:
                entries = list(target.iterdir())
            except PermissionError as exc:
                raise RuntimeError(f"Cannot read directory: {target}") from exc

            # 大小写不敏感的字母序，与官方一致；含 dotfile。
            entries.sort(key=lambda p: p.name.lower())

            rows: list[str] = []
            limit_reached = False
            for entry in entries:
                if len(rows) >= limit:
                    limit_reached = True
                    break
                try:
                    rows.append(f"{entry.name}/" if entry.is_dir() else entry.name)
                except OSError:
                    # stat 失败（断裂的符号链接等）时跳过，不让整次调用失败。
                    continue

            if not rows:
                return "(empty directory)", {"path": str(target), "count": 0}

            truncation = truncate_head("\n".join(rows), max_bytes=DEFAULT_MAX_BYTES)
            details: dict[str, Any] = {"path": str(target), "count": len(rows)}

            notices: list[str] = []
            if limit_reached:
                notices.append(
                    f"{limit} entries limit reached. Use limit={limit * 2} for more"
                )
                details["entry_limit_reached"] = limit
            if truncation.truncated:
                notices.append(f"{format_size(DEFAULT_MAX_BYTES)} limit reached")
                details["truncated"] = True

            return truncation.content + build_notice(notices), details

        text, details = await asyncio.to_thread(_work)
        return AgentToolResult(content=[TextContent(text=text)], details=details)

    return AgentTool(
        name="ls",
        label="ls",
        description=(
            "List directory contents. Returns entries sorted alphabetically, with '/' suffix "
            f"for directories. Includes dotfiles. Truncated to {DEFAULT_LIMIT} entries or "
            f"{DEFAULT_MAX_BYTES // 1024}KB, whichever is hit first."
        ),
        execute=execute,
        parameters=_SCHEMA,
        prompt_snippet=LS_PROMPT_SNIPPET,
    )
