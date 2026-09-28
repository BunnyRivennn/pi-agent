"""Storage 契约类型与 Protocol（对齐官方架构 B: session/types.ts:379-471）。

A1 落地存储层的接口面：
- 条目级/用量级写（EntryWrite / UsageWrite）与完整 Write 联合
- 扫描查询（BranchScan / EntryScan / UsageScan）与游标
- CommitResult / UsageRow / SessionStats / EntryStructure
- Storage Protocol（异步方法，与官方逐一对应）

Context 官方是依赖注入载体；A1 用最小占位 dataclass，后续切片按需扩展。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol, TypeAlias, TypeVar

from ...agent_core.types import Usage
from ..types import Entry, EntryType, JsonValue, NewEntry
from ..values import (
    ListElement,
    ListReadOptions,
    ListWrite,
    StoredValue,
    Value,
    ValueList,
    ValueWrite,
)

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Context（依赖注入载体的最小占位）
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Context:
    """贯穿存储调用的上下文。A1 为空壳，后续切片按需注入（如 clock、tracing）。"""


# ---------------------------------------------------------------------------
# 用量行 + 写操作（对齐 types.ts:379-398）
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class UsageRow:
    id: str
    seq: int
    usage: Usage
    adjustment: bool = False
    entry_id: str | None = None
    details: JsonValue = None


@dataclass(slots=True)
class EntryWrite:
    entry: NewEntry
    kind: Literal["entry"] = "entry"


@dataclass(slots=True)
class UsageWrite:
    # 对齐官方 Omit<UsageRow,"seq">：seq 由 commit 分配，这里不携带。
    id: str
    usage: Usage
    adjustment: bool = False
    entry_id: str | None = None
    details: JsonValue = None
    kind: Literal["usage"] = "usage"


# 完整 Write 联合（存储层 commit 消费）。values.Write 只含值/列表写；
# 这里并入条目级与用量级写，构成 Storage.commit 的完整入参类型。
Write: TypeAlias = EntryWrite | UsageWrite | ValueWrite | ListWrite


@dataclass(slots=True)
class CommitResult:
    first_seq: int
    seqs: list[int]
    timestamp: int
    stats: SessionStats


# ---------------------------------------------------------------------------
# 会话统计（对齐 types.ts:450）
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SessionStats:
    message_count: int = 0
    usage: Usage = field(default_factory=Usage)


# ---------------------------------------------------------------------------
# 扫描查询（对齐 types.ts:408-448）
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class EntryStructure:
    id: str
    parent_id: str | None
    seq: int
    timestamp: int
    type: EntryType
    custom_type: str | None = None


@dataclass(frozen=True, slots=True)
class EntryCursor:
    seq: int


@dataclass(slots=True)
class BranchScan:
    # start 是必填（对齐官方 StorageBranchScan = BranchScan & {start: string}）。
    start: str
    stop_at_type: EntryType | None = None
    stop_at_id: str | None = None
    type: EntryType | None = None
    custom_type: str | None = None
    order: Literal["newestFirst", "oldestFirst"] = "newestFirst"
    limit: int | None = None
    cursor: EntryCursor | None = None


@dataclass(slots=True)
class EntryScan:
    type: EntryType | None = None
    custom_type: str | None = None
    from_seq: int | None = None
    to_seq: int | None = None
    order: Literal["asc", "desc"] = "asc"
    limit: int | None = None


@dataclass(slots=True)
class UsageScan:
    from_seq: int | None = None
    to_seq: int | None = None
    order: Literal["asc", "desc"] = "asc"
    limit: int | None = None


# ---------------------------------------------------------------------------
# Storage Protocol（对齐 types.ts:455-471，逐方法对应，异步）
# ---------------------------------------------------------------------------


class Storage(Protocol):
    async def commit(
        self, writes: list[Write], context: Context
    ) -> CommitResult: ...

    async def get_entries(
        self, ids: list[str], context: Context
    ) -> dict[str, Entry]: ...

    async def get_value(
        self, address: Value[T], context: Context
    ) -> StoredValue[T] | None: ...

    async def scan_values(
        self, prefix: Value[T], context: Context
    ) -> list[StoredValue[T]]: ...

    async def read_list(
        self,
        address: ValueList[T],
        options: ListReadOptions | None,
        context: Context,
    ) -> list[ListElement[T]]: ...

    async def scan_branch(
        self, query: BranchScan, context: Context
    ) -> list[Entry]: ...

    async def scan_branch_structure(
        self, query: BranchScan, context: Context
    ) -> list[EntryStructure]: ...

    async def scan_entries(
        self, query: EntryScan, context: Context
    ) -> list[Entry]: ...

    async def scan_usage(
        self, query: UsageScan, context: Context
    ) -> list[UsageRow]: ...

    async def get_stats(self, context: Context) -> SessionStats: ...

    async def close(self, context: Context) -> None: ...


__all__ = [
    "Context",
    "UsageRow",
    "EntryWrite",
    "UsageWrite",
    "Write",
    "CommitResult",
    "SessionStats",
    "EntryStructure",
    "EntryCursor",
    "BranchScan",
    "EntryScan",
    "UsageScan",
    "Storage",
]
