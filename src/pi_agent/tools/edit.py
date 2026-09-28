"""``edit`` 工具：基于精确文本替换的文件编辑。

匹配/重叠/diff 的算法在 :mod:`.edit_diff`。本模块负责 IO、BOM 与行尾保持，
以及 :func:`prepare_edit_arguments` 那层「模型发歪了也尽量接住」的归一化。
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from ..agent_core.types import AgentTool, AgentToolResult, TextContent
from .edit_diff import (
    Edit,
    apply_edits,
    detect_line_ending,
    first_changed_line,
    generate_unified_patch,
    normalize_to_lf,
    restore_line_endings,
    split_bom,
)
from .mutation_queue import with_file_mutation_lock
from .paths import resolve_in_root
from .registry import register_tool

__all__ = [
    "create_edit_tool",
    "prepare_edit_arguments",
    "EDIT_PROMPT_SNIPPET",
    "EDIT_PROMPT_GUIDELINES",
]

EDIT_PROMPT_SNIPPET = (
    "Make precise file edits with exact text replacement, "
    "including multiple disjoint edits in one call"
)
EDIT_PROMPT_GUIDELINES = (
    "Use edit for precise changes (edits[].oldText must match exactly)",
    "When changing multiple separate locations in one file, use one edit call with multiple "
    "entries in edits[] instead of multiple edit calls",
    "Each edits[].oldText is matched against the original file, not after earlier edits are "
    "applied. Do not emit overlapping or nested edits. Merge nearby changes into one edit.",
    "Keep edits[].oldText as small as possible while still being unique in the file. "
    "Do not pad with large unchanged regions.",
)

_SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "Path to the file to edit (relative or absolute)",
        },
        "edits": {
            "type": "array",
            "description": (
                "One or more targeted replacements. Each edit is matched against the original "
                "file, not incrementally. Do not include overlapping or nested edits."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "oldText": {
                        "type": "string",
                        "description": (
                            "Exact text for one targeted replacement. It must be unique in the "
                            "original file and must not overlap with any other edits[].oldText."
                        ),
                    },
                    "newText": {
                        "type": "string",
                        "description": "Replacement text for this targeted edit.",
                    },
                },
                "required": ["oldText", "newText"],
            },
            "minItems": 1,
        },
    },
    "required": ["path", "edits"],
}


def _is_single_edit(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and isinstance(value.get("oldText"), str)
        and isinstance(value.get("newText"), str)
    )


def prepare_edit_arguments(params: Mapping[str, Any]) -> Mapping[str, Any]:
    """在 schema 校验前修正模型常见的参数形式偏差。

    实际见过的三种歪法：

    - ``edits`` 发成 JSON **字符串**而不是数组；
    - ``edits`` 发成单个对象而不是单元素数组；
    - 沿用旧接口，把 ``oldText`` / ``newText`` 平铺在顶层。

    修不动就原样返回，让 schema 校验去报错。
    """
    args = dict(params)
    edits = args.get("edits")

    if isinstance(edits, str):
        try:
            parsed = json.loads(edits)
        except (ValueError, TypeError):
            parsed = None
        if isinstance(parsed, list):
            args["edits"] = parsed
        elif _is_single_edit(parsed):
            args["edits"] = [parsed]
    elif _is_single_edit(edits):
        args["edits"] = [edits]

    # 顶层平铺的 oldText/newText 并进 edits 数组。
    old_text = args.get("oldText")
    new_text = args.get("newText")
    if isinstance(old_text, str) and isinstance(new_text, str):
        merged = list(args.get("edits") or [])
        merged.append({"oldText": old_text, "newText": new_text})
        args["edits"] = merged
        args.pop("oldText", None)
        args.pop("newText", None)

    return args


@register_tool("edit")
def create_edit_tool(
    cwd: Path,
    *,
    root: Path | None = None,
    allow_outside_root: bool = False,
) -> AgentTool:
    """构造 ``edit`` 工具。"""

    async def execute(
        tool_call_id: str,
        params: Mapping[str, Any],
        abort_event: asyncio.Event | None = None,
        on_update: Callable[[AgentToolResult[Any]], None] | None = None,
    ) -> AgentToolResult[Any]:
        del tool_call_id, on_update
        raw_path = str(params["path"])
        raw_edits = params["edits"]

        edits = [
            Edit(old_text=str(item["oldText"]), new_text=str(item["newText"]))
            for item in raw_edits
        ]

        target = resolve_in_root(raw_path, cwd, root, allow_outside_root=allow_outside_root)

        async def _do_edit() -> AgentToolResult[Any]:
            def _raise_if_aborted() -> None:
                if abort_event is not None and abort_event.is_set():
                    raise RuntimeError("Operation aborted")

            _raise_if_aborted()
            if not target.exists():
                raise RuntimeError(f"Could not edit file: {raw_path}. File does not exist.")

            raw_bytes = await asyncio.to_thread(target.read_bytes)
            _raise_if_aborted()

            try:
                raw_text = raw_bytes.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise RuntimeError(
                    f"Could not edit file: {raw_path}. It is not valid UTF-8 "
                    f"({exc.reason}). Convert it to UTF-8 first."
                ) from exc

            # 模型的 oldText 里不会有不可见 BOM，匹配前剥掉、写回时加回。
            bom, content = split_bom(raw_text)
            original_ending = detect_line_ending(content)
            normalized = normalize_to_lf(content)

            applied = apply_edits(normalized, edits, raw_path)
            _raise_if_aborted()

            final_text = bom + restore_line_endings(applied.new_content, original_ending)
            await asyncio.to_thread(target.write_text, final_text, encoding="utf-8", newline="")
            _raise_if_aborted()

            patch = generate_unified_patch(raw_path, applied.base_content, applied.new_content)
            return AgentToolResult(
                content=[
                    TextContent(
                        text=f"Successfully replaced {len(edits)} block(s) in {raw_path}."
                    )
                ],
                details={
                    "path": str(target),
                    "patch": patch,
                    "first_changed_line": first_changed_line(
                        applied.base_content, applied.new_content
                    ),
                },
            )

        return await with_file_mutation_lock(target, _do_edit)

    return AgentTool(
        name="edit",
        label="edit",
        description=(
            "Edit a single file using exact text replacement. Every edits[].oldText must match "
            "a unique, non-overlapping region of the original file. If two changes affect the "
            "same block or nearby lines, merge them into one edit instead of emitting "
            "overlapping edits."
        ),
        execute=execute,
        parameters=_SCHEMA,
        prepare_arguments=prepare_edit_arguments,
        prompt_snippet=EDIT_PROMPT_SNIPPET,
        prompt_guidelines=EDIT_PROMPT_GUIDELINES,
    )
