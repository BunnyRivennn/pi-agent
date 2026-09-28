"""提交管线：seq 分配、写校验、committed 写形态（对齐官方 commit.ts:53-116）。

commit 的原子性核心：
- ``prepare_storage_commit`` 把每个 Write 分配连续 seq（firstSeq+index）、统一 timestamp。
- ``validate_committed_writes`` 校验：seq 严格单调递增、entry/usage id 不重复、
  entry 的 parent_id 必须已存在（库中或同一事务内先出现）。
- 校验通过后由存储层原子应用；任一校验失败则整批不落库。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, Protocol, TypeAlias

from ...agent_core.types import Usage, UsageCost
from ..errors import SessionError
from ..types import Entry, EntryType, JsonValue, NewEntry
from .storage_types import EntryWrite, UsageRow, UsageWrite, Write


class CommitError(SessionError):
    """提交校验失败（seq 非单调 / id 重复 / 缺失父条目）。"""


# ---------------------------------------------------------------------------
# committed 写形态（已分配 seq/timestamp）
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class CommittedEntryWrite:
    entry: Entry  # 已 materialize（含 seq/timestamp）
    kind: Literal["entry"] = "entry"


@dataclass(slots=True)
class CommittedUsageWrite:
    row: UsageRow  # 已含 seq
    kind: Literal["usage"] = "usage"


@dataclass(slots=True)
class CommittedValueWrite:
    op: Literal["set", "delete"]
    seq: int
    namespace: str
    key: str
    value: JsonValue = None
    kind: Literal["value"] = "value"


@dataclass(slots=True)
class CommittedListWrite:
    op: Literal["append", "delete"]
    seq: int
    namespace: str
    key: str
    value: JsonValue = None
    kind: Literal["list"] = "list"


CommittedWrite: TypeAlias = (
    CommittedEntryWrite
    | CommittedUsageWrite
    | CommittedValueWrite
    | CommittedListWrite
)


@dataclass(slots=True)
class PreparedCommit:
    writes: list[CommittedWrite]
    first_seq: int
    seqs: list[int]
    timestamp: int


# ---------------------------------------------------------------------------
# Write 构造辅助（对齐 insertEntry / insertUsage）
# ---------------------------------------------------------------------------


def insert_entry(entry: NewEntry) -> EntryWrite:
    return EntryWrite(entry=entry)


def insert_usage(
    id: str,
    usage: Usage,
    adjustment: bool = False,
    entry_id: str | None = None,
    details: JsonValue = None,
) -> UsageWrite:
    return UsageWrite(
        id=id,
        usage=usage,
        adjustment=adjustment,
        entry_id=entry_id,
        details=details,
    )


# ---------------------------------------------------------------------------
# seq 分配 + materialize
# ---------------------------------------------------------------------------


def _materialize_entry(entry: NewEntry, seq: int, timestamp: int) -> Entry:
    # NewEntry 即 Entry（哨兵 seq/timestamp）；分配真实值后返回副本。
    return replace(entry, seq=seq, timestamp=timestamp)


def commit_write(write: Write, seq: int, timestamp: int) -> CommittedWrite:
    if write.kind == "entry":
        return CommittedEntryWrite(entry=_materialize_entry(write.entry, seq, timestamp))
    if write.kind == "usage":
        return CommittedUsageWrite(
            row=UsageRow(
                id=write.id,
                seq=seq,
                usage=write.usage,
                adjustment=write.adjustment,
                entry_id=write.entry_id,
                details=write.details,
            )
        )
    if write.kind == "value":
        return CommittedValueWrite(
            op=write.op,
            seq=seq,
            namespace=write.namespace,
            key=write.key,
            value=getattr(write, "value", None),
        )
    if write.kind == "list":
        return CommittedListWrite(
            op=write.op,
            seq=seq,
            namespace=write.namespace,
            key=write.key,
            value=getattr(write, "value", None),
        )
    raise CommitError(f"unknown write kind: {write!r}")


def prepare_storage_commit(
    writes: list[Write], first_seq: int, timestamp: int
) -> PreparedCommit:
    committed = [
        commit_write(w, first_seq + index, timestamp) for index, w in enumerate(writes)
    ]
    return PreparedCommit(
        writes=committed,
        first_seq=first_seq,
        seqs=[committed_seq(w) for w in committed],
        timestamp=timestamp,
    )


def committed_seq(write: CommittedWrite) -> int:
    if isinstance(write, CommittedEntryWrite):
        return write.entry.seq
    if isinstance(write, CommittedUsageWrite):
        return write.row.seq
    return write.seq


# ---------------------------------------------------------------------------
# 校验
# ---------------------------------------------------------------------------


class CommitValidationState(Protocol):
    def has_entry_or_usage_id(self, id: str) -> bool: ...
    def has_entry_id(self, id: str) -> bool: ...


def _committed_id_and_type(
    write: CommittedWrite,
) -> tuple[str, EntryType | None, str | None] | None:
    """返回 (id, entry_type_if_entry, parent_id_if_entry) 供校验；非 entry/usage 返回 None。"""
    if isinstance(write, CommittedEntryWrite):
        return write.entry.id, write.entry.type, write.entry.parent_id
    if isinstance(write, CommittedUsageWrite):
        return write.row.id, None, None
    return None


def validate_committed_writes(
    writes: list[CommittedWrite],
    first_seq: int,
    state: CommitValidationState,
) -> None:
    previous_seq = first_seq - 1
    transaction_ids: set[str] = set()
    transaction_entry_ids: set[str] = set()
    for write in writes:
        seq = committed_seq(write)
        if seq <= previous_seq:
            raise CommitError(f"Non-monotonic storage sequence: {seq}")
        previous_seq = seq

        info = _committed_id_and_type(write)
        if info is None:
            continue
        write_id, entry_type, parent_id = info
        if state.has_entry_or_usage_id(write_id) or write_id in transaction_ids:
            raise CommitError(f"Duplicate entry or usage id: {write_id}")
        if (
            entry_type is not None
            and parent_id is not None
            and not state.has_entry_id(parent_id)
            and parent_id not in transaction_entry_ids
        ):
            raise CommitError(f"Missing parent entry: {parent_id}")
        transaction_ids.add(write_id)
        if entry_type is not None:
            transaction_entry_ids.add(write_id)


# ---------------------------------------------------------------------------
# Usage 合并（对齐 utils/usage.ts addUsage；SessionStats.usage 累加用）
# ---------------------------------------------------------------------------


def add_usage(left: Usage, right: Usage) -> Usage:
    return Usage(
        input=left.input + right.input,
        output=left.output + right.output,
        cache_read=left.cache_read + right.cache_read,
        cache_write=left.cache_write + right.cache_write,
        total_tokens=left.total_tokens + right.total_tokens,
        cost=UsageCost(
            input=left.cost.input + right.cost.input,
            output=left.cost.output + right.cost.output,
            cache_read=left.cost.cache_read + right.cost.cache_read,
            cache_write=left.cost.cache_write + right.cost.cache_write,
            total=left.cost.total + right.cost.total,
        ),
    )


__all__ = [
    "CommitError",
    "CommittedEntryWrite",
    "CommittedUsageWrite",
    "CommittedValueWrite",
    "CommittedListWrite",
    "CommittedWrite",
    "PreparedCommit",
    "insert_entry",
    "insert_usage",
    "commit_write",
    "prepare_storage_commit",
    "committed_seq",
    "CommitValidationState",
    "validate_committed_writes",
    "add_usage",
]
