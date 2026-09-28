"""短期记忆 · 压缩纯函数层（D1）。

对齐官方 harness ``compaction`` 实现，只含无 I/O 的纯函数：token 估算、
压缩阈值判定、切点选择、会话文本序列化。LLM 摘要调用与 entry 落库属于 D2。

与官方的两处有意偏离（均在保守方向）：
1. **CJK 感知的 token 估算**：官方一律 ``ceil(chars/4)``，对中文严重低估
   （一个汉字约 1 token，被算成 0.25）。本项目注释/会话大量使用中文，估算偏低
   会导致真正超窗后才触发压缩、被模型拒绝。这里 CJK 字符按 1 token 计，其余
   仍按 /4。宁可偏早压缩，不冒超窗风险。
2. **消息角色收窄**：官方 ``AgentMessage`` 含 bashExecution/custom/branchSummary/
   compactionSummary 等角色；本项目只有 user/assistant/toolResult 三种具体消息，
   故 estimate/序列化分支相应收窄。

跳过官方 coding-agent 专属的文件追踪（read/write/edit 工具路径提取）：它把工具名
写死在压缩模块里，属于编码 agent 的领域知识，与通用 agent 框架无关。
"""

from __future__ import annotations

import json
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

from .types import (
    AgentMessage,
    AssistantMessage,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)

# 摘要 prompt 里单条 tool result 的最大字符数，超出截断。
TOOL_RESULT_MAX_CHARS = 2000
# 图片内容按固定字符数估算（对齐官方 ESTIMATED_IMAGE_CHARS）。
_ESTIMATED_IMAGE_CHARS = 4800
# 不参与 token 计数（无有效 usage）的 assistant 停止原因。
_INVALID_USAGE_STOP_REASONS = frozenset({"aborted", "error"})


@dataclass(slots=True)
class CompactionSettings:
    """压缩阈值与保留设置。"""

    enabled: bool = True
    # 为摘要 prompt + 模型输出预留的 token。
    reserve_tokens: int = 16384
    # 压缩后大致保留的近期上下文 token。
    keep_recent_tokens: int = 20000


DEFAULT_COMPACTION_SETTINGS = CompactionSettings()


# ---------------------------------------------------------------------------
# 文本提取
# ---------------------------------------------------------------------------


def content_text(
    content: str | Sequence[object],
    default: str = "",
) -> str:
    """把消息 content（字符串或 content block 列表）拍平成纯文本。

    只取 TextContent 的 text；其余 block（图片/thinking/toolCall）忽略。
    对齐官方 pi-ai 的 ``contentText``。
    """
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        text = getattr(block, "text", None)
        if isinstance(text, str) and text:
            parts.append(text)
    joined = "".join(parts)
    return joined if joined else default


def _safe_json(value: object) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return "[unserializable]"


# ---------------------------------------------------------------------------
# token 估算（CJK 感知）
# ---------------------------------------------------------------------------


def _is_cjk(ch: str) -> bool:
    """判断字符是否属于 CJK（含中日韩表意文字、假名、谚文等宽字符）。

    这些字符普遍映射为约 1 个 token，不能按拉丁文的 chars/4 估算。
    用 East Asian Width == 'W'/'F'（宽/全角）覆盖主流 CJK。
    """
    return unicodedata.east_asian_width(ch) in ("W", "F")


def _estimate_chars_as_tokens(text: str) -> int:
    """把一段文本估算成 token 数：CJK 字符按 1，其余累计后按 ceil(/4)。"""
    cjk = 0
    other = 0
    for ch in text:
        if _is_cjk(ch):
            cjk += 1
        else:
            other += 1
    return cjk + -(-other // 4)  # -(-x//4) == ceil(x/4)


def _text_and_image_chars_as_tokens(
    content: str | Sequence[object],
) -> int:
    if isinstance(content, str):
        return _estimate_chars_as_tokens(content)
    total = 0
    for block in content:
        btype = getattr(block, "type", None)
        if btype == "text":
            text = getattr(block, "text", "")
            if isinstance(text, str):
                total += _estimate_chars_as_tokens(text)
        elif btype == "image":
            total += -(-_ESTIMATED_IMAGE_CHARS // 4)
    return total


def estimate_tokens(message: AgentMessage) -> int:
    """用保守字符启发式估算单条消息的 token 数（CJK 感知）。"""
    role = getattr(message, "role", None)
    if role == "user" and isinstance(message, UserMessage):
        return _text_and_image_chars_as_tokens(message.content)
    if role == "assistant" and isinstance(message, AssistantMessage):
        total = 0
        for block in message.content:
            if isinstance(block, TextContent):
                total += _estimate_chars_as_tokens(block.text)
            elif isinstance(block, ThinkingContent):
                total += _estimate_chars_as_tokens(block.thinking)
            elif isinstance(block, ToolCall):
                total += _estimate_chars_as_tokens(
                    block.name + _safe_json(block.arguments)
                )
        return total
    if role == "toolResult" and isinstance(message, ToolResultMessage):
        return _text_and_image_chars_as_tokens(message.content)
    return 0


# ---------------------------------------------------------------------------
# usage 汇总与压缩阈值
# ---------------------------------------------------------------------------


def calculate_context_tokens(usage: Usage) -> int:
    """从 provider usage 计算上下文 token 总数。"""
    if usage.total_tokens:
        return usage.total_tokens
    return usage.input + usage.output + usage.cache_read + usage.cache_write


def _assistant_usage(message: AgentMessage) -> Usage | None:
    """取有效 assistant usage（非 aborted/error 且 token 数 > 0）。"""
    if not isinstance(message, AssistantMessage):
        return None
    if message.stop_reason in _INVALID_USAGE_STOP_REASONS:
        return None
    if calculate_context_tokens(message.usage) > 0:
        return message.usage
    return None


@dataclass(slots=True)
class ContextUsageEstimate:
    """消息列表的上下文 token 估算结果。"""

    tokens: int
    usage_tokens: int
    trailing_tokens: int
    last_usage_index: int | None


def estimate_context_tokens(messages: Sequence[AgentMessage]) -> ContextUsageEstimate:
    """估算一组消息的上下文 token：优先用最近一条有效 assistant usage，
    加上其后消息的估算尾部；无 usage 时全部估算。"""
    last_index: int | None = None
    last_usage: Usage | None = None
    for i in range(len(messages) - 1, -1, -1):
        usage = _assistant_usage(messages[i])
        if usage is not None:
            last_index = i
            last_usage = usage
            break

    if last_usage is None or last_index is None:
        estimated = sum(estimate_tokens(m) for m in messages)
        return ContextUsageEstimate(
            tokens=estimated,
            usage_tokens=0,
            trailing_tokens=estimated,
            last_usage_index=None,
        )

    usage_tokens = calculate_context_tokens(last_usage)
    trailing = sum(estimate_tokens(m) for m in messages[last_index + 1 :])
    return ContextUsageEstimate(
        tokens=usage_tokens + trailing,
        usage_tokens=usage_tokens,
        trailing_tokens=trailing,
        last_usage_index=last_index,
    )


def should_compact(
    context_tokens: int, context_window: int, settings: CompactionSettings
) -> bool:
    """上下文用量是否超过压缩阈值（窗口 - 预留）。"""
    if not settings.enabled:
        return False
    return context_tokens > context_window - settings.reserve_tokens


# ---------------------------------------------------------------------------
# 会话文本序列化（供摘要 prompt 使用）
# ---------------------------------------------------------------------------


def _truncate_for_summary(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    dropped = len(text) - max_chars
    return f"{text[:max_chars]}\n\n[... {dropped} more characters truncated]"


def serialize_conversation(messages: Sequence[AgentMessage]) -> str:
    """把 LLM 消息序列化成纯文本，喂给摘要模型。

    分段：[User] / [Assistant thinking] / [Assistant] / [Assistant tool calls] /
    [Tool result]（后者按 TOOL_RESULT_MAX_CHARS 截断）。
    """
    parts: list[str] = []
    for msg in messages:
        role = getattr(msg, "role", None)
        if role == "user" and isinstance(msg, UserMessage):
            text = content_text(msg.content, "")
            if text:
                parts.append(f"[User]: {text}")
        elif role == "assistant" and isinstance(msg, AssistantMessage):
            thinking: list[str] = []
            tool_calls: list[str] = []
            has_text = False
            for block in msg.content:
                if isinstance(block, ThinkingContent):
                    thinking.append(block.thinking)
                elif isinstance(block, ToolCall):
                    args = ", ".join(
                        f"{k}={_safe_json(v)}" for k, v in block.arguments.items()
                    )
                    tool_calls.append(f"{block.name}({args})")
                elif isinstance(block, TextContent):
                    has_text = True
            if thinking:
                parts.append("[Assistant thinking]: " + "\n".join(thinking))
            if has_text:
                parts.append(f"[Assistant]: {content_text(msg.content)}")
            if tool_calls:
                parts.append("[Assistant tool calls]: " + "; ".join(tool_calls))
        elif role == "toolResult" and isinstance(msg, ToolResultMessage):
            text = content_text(msg.content, "")
            if text:
                parts.append(
                    f"[Tool result]: {_truncate_for_summary(text, TOOL_RESULT_MAX_CHARS)}"
                )
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# 摘要 prompt 常量
# ---------------------------------------------------------------------------

SUMMARIZATION_SYSTEM_PROMPT = (
    "You are a context summarization assistant. Your task is to read a "
    "conversation between a user and an AI assistant, then produce a structured "
    "summary following the exact format specified.\n\n"
    "Do NOT continue the conversation. Do NOT respond to any questions in the "
    "conversation. ONLY output the structured summary."
)

SUMMARIZATION_PROMPT = """The messages above are a conversation to summarize. \
Create a structured context checkpoint summary that another LLM will use to \
continue the work.

Use this EXACT format:

## Goal
[What is the user trying to accomplish?]

## Constraints & Preferences
- [Any constraints, preferences, or requirements mentioned by user]
- [Or "(none)" if none were mentioned]

## Progress
### Done
- [x] [Completed tasks/changes]

### In Progress
- [ ] [Current work]

### Blocked
- [ ] [Anything blocked and why, or "(none)"]

## Key Facts
- [Important facts, decisions, file paths, identifiers discovered]

## Next Steps
- [What should happen next]
"""


__all__ = [
    "TOOL_RESULT_MAX_CHARS",
    "CompactionSettings",
    "DEFAULT_COMPACTION_SETTINGS",
    "ContextUsageEstimate",
    "content_text",
    "estimate_tokens",
    "estimate_context_tokens",
    "calculate_context_tokens",
    "should_compact",
    "serialize_conversation",
    "SUMMARIZATION_SYSTEM_PROMPT",
    "SUMMARIZATION_PROMPT",
]
