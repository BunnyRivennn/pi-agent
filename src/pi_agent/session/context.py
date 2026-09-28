"""压缩感知的上下文构建（对齐官方 session/context.ts）。

三个纯函数：
- ``build_context_entries``：沿 branch 路径找最后一个 compaction，截断为
  ``[compaction, ...compaction 之后的 entry]``；无 compaction 则原样返回。
- ``session_entry_to_context_messages``：把单条 entry 投影为发给 LLM 的消息。
- ``build_session_context``：组合上二者，输出完整上下文消息列表。

与官方的有意偏离：官方有独立的 ``compactionSummary`` / ``branchSummary`` 消息 role；
我们的 AgentMessage 只有 user/assistant/toolResult（不改动 agent_core 消息模型），
故把摘要投影成带前缀标记的 UserMessage。
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Sequence
from typing import TypeAlias, Union

from ..agent_core.types import AgentMessage, AssistantMessage, UserMessage
from .types import Entry

# custom entry 投影器：决定某 customType 是否进上下文、投影成什么消息。
_ProjectorResult: TypeAlias = Union[
    "list[AgentMessage] | None", Awaitable["list[AgentMessage] | None"]
]
EntryProjector: TypeAlias = Callable[[Entry], _ProjectorResult]

_COMPACTION_PREFIX = "[Conversation Summary]"
_BRANCH_PREFIX = "[Branch Summary]"

# assistant 消息中这些终止原因不进上下文（未完成/出错的轮次）。
# 注：官方还有 "deferred"，我们的 StopReason 模型无此值，故不含。
_EXCLUDED_STOP_REASONS = frozenset({"error", "aborted"})


def _compaction_summary_message(summary: str, tokens_before: int, timestamp: int) -> UserMessage:
    text = f"{_COMPACTION_PREFIX} (compacted {tokens_before} tokens)\n{summary}"
    return UserMessage(content=text, timestamp=timestamp)


def _branch_summary_message(summary: str, from_id: str | None, timestamp: int) -> UserMessage:
    text = f"{_BRANCH_PREFIX} (from {from_id})\n{summary}"
    return UserMessage(content=text, timestamp=timestamp)


def is_context_message(message: AgentMessage) -> bool:
    if not isinstance(message, AssistantMessage):
        return True
    return message.stop_reason not in _EXCLUDED_STOP_REASONS


def build_context_entries(path_entries: Sequence[Entry]) -> list[Entry]:
    compaction_index = -1
    for index in range(len(path_entries) - 1, -1, -1):
        if path_entries[index].type == "compaction":
            compaction_index = index
            break
    if compaction_index == -1:
        return list(path_entries)
    return list(path_entries[compaction_index:])


def session_entry_to_context_messages(entry: Entry) -> list[AgentMessage]:
    if entry.type == "message":
        msg = entry.message
        if msg is None:
            return []
        return [msg] if is_context_message(msg) else []
    if entry.type == "compaction":
        tail = [m for m in entry.retained_tail if is_context_message(m)]
        return [
            _compaction_summary_message(
                entry.summary, entry.tokens_before, entry.timestamp
            ),
            *tail,
        ]
    if entry.type == "branch_summary":
        if not entry.summary:
            return []
        return [
            _branch_summary_message(entry.summary, entry.from_id, entry.timestamp)
        ]
    # custom：默认不进上下文（由 projector 决定，见 build_session_context）。
    return []


async def build_session_context(
    path_entries: Sequence[Entry],
    projectors: dict[str, EntryProjector] | None = None,
) -> list[AgentMessage]:
    projectors = projectors or {}
    entries = build_context_entries(path_entries)
    messages: list[AgentMessage] = []
    for entry in entries:
        if entry.type != "custom":
            messages.extend(session_entry_to_context_messages(entry))
            continue
        projector = projectors.get(entry.custom_type or "")
        if projector is None:
            continue
        result = projector(entry)
        if inspect.isawaitable(result):
            result = await result
        if result:
            messages.extend(result)
    return messages


__all__ = [
    "EntryProjector",
    "is_context_message",
    "build_context_entries",
    "session_entry_to_context_messages",
    "build_session_context",
]
