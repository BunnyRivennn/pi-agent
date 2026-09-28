"""会话持久化的核心数据模型（对齐官方架构 B: packages/agent/src/harness/session/types.ts）。

四种 Entry 构成 append-only 会话树：
- MessageEntry       : 一条对话消息
- CompactionEntry    : 压缩摘要（retainedTail 内联保留近期消息）
- BranchSummaryEntry : 分叉摘要
- CustomEntry        : 扩展自定义条目

设计约束：
- ``seq`` / ``timestamp`` 由存储层（Storage）在 commit 时分配，因此对外构造用
  ``NewEntry``（不含这两个字段）。
- ``id`` / ``parent_id`` 构成树；``parent_id is None`` 表示根节点。
- 消息本体沿用 ``agent_core`` 的 dataclass，本模块不重新定义消息。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias

from ..agent_core.types import AgentMessage

# ---------------------------------------------------------------------------
# Entry 类型
# ---------------------------------------------------------------------------

EntryType = Literal["message", "compaction", "branch_summary", "custom"]

# 侧存储 JSON 值：官方用 JsonValue，这里用递归别名表达可 JSON 序列化的值。
JsonValue: TypeAlias = (
    None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
)


@dataclass(slots=True)
class EntryBase:
    """所有 Entry 的公共字段。

    ``seq`` / ``timestamp`` 由存储层分配；构造新条目时用 ``NewEntry``（见文件末）。
    """

    id: str
    parent_id: str | None
    # 哨兵默认：seq=-1 / timestamp=0 表示"尚未分配"，由存储层 commit 时覆盖。
    # 这样对外构造条目（NewEntry 语义）可省略这两个字段。
    seq: int = -1
    timestamp: int = 0
    type: EntryType = "message"
    custom_type: str | None = None


@dataclass(slots=True)
class MessageEntry(EntryBase):
    type: Literal["message"] = "message"
    message: AgentMessage | None = None
    # 官方 terminate?: true —— 标记该消息终止一个 turn。仅 True 或不存在。
    terminate: bool | None = None


@dataclass(slots=True)
class CompactionEntry(EntryBase):
    type: Literal["compaction"] = "compaction"
    summary: str = ""
    # 压缩时保留的近期消息（内联，自包含；不用 firstKeptEntryId 跨条查找）。
    retained_tail: list[AgentMessage] = field(default_factory=list)
    tokens_before: int = 0
    details: JsonValue = None
    usage: dict[str, Any] | None = None
    # True = 扩展生成；False = pi 内建生成。
    from_hook: bool = False


@dataclass(slots=True)
class BranchSummaryEntry(EntryBase):
    type: Literal["branch_summary"] = "branch_summary"
    from_id: str | None = None
    summary: str = ""
    details: JsonValue = None
    usage: dict[str, Any] | None = None
    from_hook: bool = False


@dataclass(slots=True)
class CustomEntry(EntryBase):
    type: Literal["custom"] = "custom"
    # custom 条目必须带 custom_type（在 EntryBase 中定义，此处语义要求非 None）。
    data: JsonValue = None


Entry: TypeAlias = MessageEntry | CompactionEntry | BranchSummaryEntry | CustomEntry

# 待写入条目（对齐官方 NewEntry = Omit<Entry, "seq"|"timestamp">）。
# Python 无结构化 Omit：这里用 Entry 的哨兵默认（seq=-1 / timestamp=0）表达
# "尚未分配序号与时间戳"，构造时不传这两个字段即可。存储层在 commit 时分配真实值。
NewEntry: TypeAlias = Entry


__all__ = [
    "EntryType",
    "JsonValue",
    "EntryBase",
    "MessageEntry",
    "CompactionEntry",
    "BranchSummaryEntry",
    "CustomEntry",
    "Entry",
    "NewEntry",
]
