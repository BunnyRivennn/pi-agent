"""dataclass ⇄ dict 的 roundtrip 序列化（A0 核心，纯函数）。

覆盖：
- content block：TextContent / ThinkingContent / ImageContent / ToolCall
- 消息：UserMessage / AssistantMessage / ToolResultMessage
- Usage / UsageCost
- 4 种 Entry：MessageEntry / CompactionEntry / BranchSummaryEntry / CustomEntry

容错原则（对齐 PLAN4 §6 红线）：未知 type / 缺失判别字段一律显式抛
``SessionCorruptError``，绝不静默丢字段或猜测。
"""

from __future__ import annotations

from typing import Any

from ..agent_core.types import (
    AgentMessage,
    AssistantContentBlock,
    AssistantMessage,
    ImageContent,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultContentBlock,
    ToolResultMessage,
    Usage,
    UsageCost,
    UserContentBlock,
    UserMessage,
)
from .errors import SessionCorruptError
from .types import (
    BranchSummaryEntry,
    CompactionEntry,
    CustomEntry,
    Entry,
    MessageEntry,
)

# ---------------------------------------------------------------------------
# content block
# ---------------------------------------------------------------------------


def _require(mapping: dict[str, Any], key: str, ctx: str) -> Any:
    if key not in mapping:
        raise SessionCorruptError(f"{ctx}: missing required field {key!r}")
    return mapping[key]


def content_block_to_dict(
    block: UserContentBlock | AssistantContentBlock | ToolResultContentBlock,
) -> dict[str, Any]:
    if isinstance(block, TextContent):
        out: dict[str, Any] = {"type": "text", "text": block.text}
        if block.text_signature is not None:
            out["text_signature"] = block.text_signature
        return out
    if isinstance(block, ThinkingContent):
        out = {"type": "thinking", "thinking": block.thinking}
        if block.thinking_signature is not None:
            out["thinking_signature"] = block.thinking_signature
        return out
    if isinstance(block, ImageContent):
        return {"type": "image", "data": block.data, "mime_type": block.mime_type}
    if isinstance(block, ToolCall):
        out = {
            "type": "toolCall",
            "id": block.id,
            "name": block.name,
            "arguments": block.arguments,
        }
        if block.thought_signature is not None:
            out["thought_signature"] = block.thought_signature
        return out
    raise SessionCorruptError(f"unknown content block type: {type(block).__name__}")


def content_block_from_dict(
    data: dict[str, Any],
) -> TextContent | ThinkingContent | ImageContent | ToolCall:
    block_type = _require(data, "type", "content block")
    if block_type == "text":
        return TextContent(
            text=_require(data, "text", "text block"),
            text_signature=data.get("text_signature"),
        )
    if block_type == "thinking":
        return ThinkingContent(
            thinking=_require(data, "thinking", "thinking block"),
            thinking_signature=data.get("thinking_signature"),
        )
    if block_type == "image":
        return ImageContent(
            data=_require(data, "data", "image block"),
            mime_type=_require(data, "mime_type", "image block"),
        )
    if block_type == "toolCall":
        return ToolCall(
            id=_require(data, "id", "toolCall block"),
            name=_require(data, "name", "toolCall block"),
            arguments=_require(data, "arguments", "toolCall block"),
            thought_signature=data.get("thought_signature"),
        )
    raise SessionCorruptError(f"unknown content block type: {block_type!r}")


def _content_to_json(
    content: str | list[Any],
) -> str | list[dict[str, Any]]:
    if isinstance(content, str):
        return content
    return [content_block_to_dict(b) for b in content]


def _user_content_from_json(content: Any) -> str | list[UserContentBlock]:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        blocks: list[UserContentBlock] = []
        for b in content:
            block = content_block_from_dict(b)
            if not isinstance(block, (TextContent, ImageContent)):
                raise SessionCorruptError(
                    f"invalid user content block: {block.type}"
                )
            blocks.append(block)
        return blocks
    raise SessionCorruptError(f"invalid message content: {type(content).__name__}")


def _assistant_content_from_json(content: Any) -> list[AssistantContentBlock]:
    if not isinstance(content, list):
        raise SessionCorruptError(
            f"assistant content must be a list, got {type(content).__name__}"
        )
    blocks: list[AssistantContentBlock] = []
    for b in content:
        block = content_block_from_dict(b)
        if not isinstance(block, (TextContent, ThinkingContent, ToolCall)):
            raise SessionCorruptError(f"invalid assistant content block: {block.type}")
        blocks.append(block)
    return blocks


def _tool_result_content_from_json(content: Any) -> list[ToolResultContentBlock]:
    if not isinstance(content, list):
        raise SessionCorruptError(
            f"toolResult content must be a list, got {type(content).__name__}"
        )
    blocks: list[ToolResultContentBlock] = []
    for b in content:
        block = content_block_from_dict(b)
        if not isinstance(block, (TextContent, ImageContent)):
            raise SessionCorruptError(
                f"invalid toolResult content block: {block.type}"
            )
        blocks.append(block)
    return blocks


# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------


def usage_to_dict(usage: Usage) -> dict[str, Any]:
    c = usage.cost
    return {
        "input": usage.input,
        "output": usage.output,
        "cache_read": usage.cache_read,
        "cache_write": usage.cache_write,
        "total_tokens": usage.total_tokens,
        "cost": {
            "input": c.input,
            "output": c.output,
            "cache_read": c.cache_read,
            "cache_write": c.cache_write,
            "total": c.total,
        },
    }


def usage_from_dict(data: dict[str, Any]) -> Usage:
    cost_data = data.get("cost", {})
    return Usage(
        input=data.get("input", 0),
        output=data.get("output", 0),
        cache_read=data.get("cache_read", 0),
        cache_write=data.get("cache_write", 0),
        total_tokens=data.get("total_tokens", 0),
        cost=UsageCost(
            input=cost_data.get("input", 0.0),
            output=cost_data.get("output", 0.0),
            cache_read=cost_data.get("cache_read", 0.0),
            cache_write=cost_data.get("cache_write", 0.0),
            total=cost_data.get("total", 0.0),
        ),
    )


# ---------------------------------------------------------------------------
# 消息
# ---------------------------------------------------------------------------


def message_to_dict(message: AgentMessage) -> dict[str, Any]:
    if isinstance(message, UserMessage):
        return {
            "role": "user",
            "content": _content_to_json(message.content),
            "timestamp": message.timestamp,
        }
    if isinstance(message, AssistantMessage):
        return {
            "role": "assistant",
            "content": _content_to_json(message.content),
            "api": message.api,
            "provider": message.provider,
            "model": message.model,
            "usage": usage_to_dict(message.usage),
            "stop_reason": message.stop_reason,
            "error_message": message.error_message,
            "timestamp": message.timestamp,
        }
    if isinstance(message, ToolResultMessage):
        return {
            "role": "toolResult",
            "tool_call_id": message.tool_call_id,
            "tool_name": message.tool_name,
            "content": _content_to_json(message.content),
            "is_error": message.is_error,
            "details": message.details,
            "timestamp": message.timestamp,
        }
    raise SessionCorruptError(
        f"cannot serialize non-LLM message: {type(message).__name__}"
    )


def message_from_dict(data: dict[str, Any]) -> AgentMessage:
    role = _require(data, "role", "message")
    if role == "user":
        return UserMessage(
            content=_user_content_from_json(_require(data, "content", "user message")),
            timestamp=data.get("timestamp", 0),
        )
    if role == "assistant":
        return AssistantMessage(
            content=_assistant_content_from_json(
                _require(data, "content", "assistant message")
            ),
            api=_require(data, "api", "assistant message"),
            provider=_require(data, "provider", "assistant message"),
            model=_require(data, "model", "assistant message"),
            usage=usage_from_dict(data.get("usage", {})),
            stop_reason=data.get("stop_reason", "stop"),
            error_message=data.get("error_message"),
            timestamp=data.get("timestamp", 0),
        )
    if role == "toolResult":
        return ToolResultMessage(
            tool_call_id=_require(data, "tool_call_id", "toolResult message"),
            tool_name=_require(data, "tool_name", "toolResult message"),
            content=_tool_result_content_from_json(
                _require(data, "content", "toolResult message")
            ),
            is_error=data.get("is_error", False),
            details=data.get("details"),
            timestamp=data.get("timestamp", 0),
        )
    raise SessionCorruptError(f"unknown message role: {role!r}")


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------


def _base_to_dict(entry: Entry) -> dict[str, Any]:
    out: dict[str, Any] = {
        "id": entry.id,
        "parent_id": entry.parent_id,
        "seq": entry.seq,
        "timestamp": entry.timestamp,
        "type": entry.type,
    }
    if entry.custom_type is not None:
        out["custom_type"] = entry.custom_type
    return out


def entry_to_dict(entry: Entry) -> dict[str, Any]:
    out = _base_to_dict(entry)
    if isinstance(entry, MessageEntry):
        out["message"] = (
            message_to_dict(entry.message) if entry.message is not None else None
        )
        if entry.terminate is not None:
            out["terminate"] = entry.terminate
        return out
    if isinstance(entry, CompactionEntry):
        out.update(
            {
                "summary": entry.summary,
                "retained_tail": [message_to_dict(m) for m in entry.retained_tail],
                "tokens_before": entry.tokens_before,
                "details": entry.details,
                "usage": entry.usage,
                "from_hook": entry.from_hook,
            }
        )
        return out
    if isinstance(entry, BranchSummaryEntry):
        out.update(
            {
                "from_id": entry.from_id,
                "summary": entry.summary,
                "details": entry.details,
                "usage": entry.usage,
                "from_hook": entry.from_hook,
            }
        )
        return out
    if isinstance(entry, CustomEntry):
        out["data"] = entry.data
        return out
    raise SessionCorruptError(f"unknown entry type: {type(entry).__name__}")


def _base_fields(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": _require(data, "id", "entry"),
        "parent_id": _require(data, "parent_id", "entry"),
        "seq": _require(data, "seq", "entry"),
        "timestamp": _require(data, "timestamp", "entry"),
        "custom_type": data.get("custom_type"),
    }


def entry_from_dict(data: dict[str, Any]) -> Entry:
    entry_type = _require(data, "type", "entry")
    base = _base_fields(data)
    if entry_type == "message":
        raw = data.get("message")
        return MessageEntry(
            **base,
            message=message_from_dict(raw) if raw is not None else None,
            terminate=data.get("terminate"),
        )
    if entry_type == "compaction":
        return CompactionEntry(
            **base,
            summary=data.get("summary", ""),
            retained_tail=[
                message_from_dict(m) for m in data.get("retained_tail", [])
            ],
            tokens_before=data.get("tokens_before", 0),
            details=data.get("details"),
            usage=data.get("usage"),
            from_hook=data.get("from_hook", False),
        )
    if entry_type == "branch_summary":
        return BranchSummaryEntry(
            **base,
            from_id=data.get("from_id"),
            summary=data.get("summary", ""),
            details=data.get("details"),
            usage=data.get("usage"),
            from_hook=data.get("from_hook", False),
        )
    if entry_type == "custom":
        return CustomEntry(**base, data=data.get("data"))
    raise SessionCorruptError(f"unknown entry type: {entry_type!r}")


__all__ = [
    "content_block_to_dict",
    "content_block_from_dict",
    "usage_to_dict",
    "usage_from_dict",
    "message_to_dict",
    "message_from_dict",
    "entry_to_dict",
    "entry_from_dict",
]
