"""``write`` 工具：写入/覆盖文件，自动建父目录。"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from ..agent_core.types import AgentTool, AgentToolResult, TextContent
from .mutation_queue import with_file_mutation_lock
from .paths import resolve_in_root
from .registry import register_tool

__all__ = ["create_write_tool", "WRITE_PROMPT_SNIPPET", "WRITE_PROMPT_GUIDELINES"]

WRITE_PROMPT_SNIPPET = "Create or overwrite files"
WRITE_PROMPT_GUIDELINES = ("Use write only for new files or complete rewrites.",)

_SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "Path to the file to write (relative or absolute)",
        },
        "content": {
            "type": "string",
            "description": "Content to write to the file",
        },
    },
    "required": ["path", "content"],
}


@register_tool("write")
def create_write_tool(
    cwd: Path,
    *,
    root: Path | None = None,
    allow_outside_root: bool = False,
) -> AgentTool:
    """构造 ``write`` 工具。"""

    async def execute(
        tool_call_id: str,
        params: Mapping[str, Any],
        abort_event: asyncio.Event | None = None,
        on_update: Callable[[AgentToolResult[Any]], None] | None = None,
    ) -> AgentToolResult[Any]:
        del tool_call_id, on_update
        raw_path = str(params["path"])
        content = str(params["content"])

        target = resolve_in_root(raw_path, cwd, root, allow_outside_root=allow_outside_root)

        async def _do_write() -> AgentToolResult[Any]:
            # 不在 abort 监听里提前返回：那会在真实 IO 还没落地时就放掉写锁。
            # 改为每个 await 之后检查，既能观察到中断，又保证锁持有到操作结束。
            def _raise_if_aborted() -> None:
                if abort_event is not None and abort_event.is_set():
                    raise RuntimeError("Operation aborted")

            _raise_if_aborted()
            await asyncio.to_thread(target.parent.mkdir, parents=True, exist_ok=True)
            _raise_if_aborted()
            await asyncio.to_thread(target.write_text, content, encoding="utf-8")
            _raise_if_aborted()

            return AgentToolResult(
                content=[TextContent(text=f"Successfully wrote to {raw_path}")],
                details={"path": str(target), "bytes": len(content.encode("utf-8"))},
            )

        return await with_file_mutation_lock(target, _do_write)

    return AgentTool(
        name="write",
        label="write",
        description=(
            "Write content to a file. Creates the file if it doesn't exist, overwrites if it "
            "does. Automatically creates parent directories."
        ),
        execute=execute,
        parameters=_SCHEMA,
        prompt_snippet=WRITE_PROMPT_SNIPPET,
        prompt_guidelines=WRITE_PROMPT_GUIDELINES,
    )
