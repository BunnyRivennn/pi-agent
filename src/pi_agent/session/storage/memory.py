"""内存存储后端（对齐官方 in-memory-storage-state.ts + memory.ts）。

- ``InMemoryStorageState``：完整物化会话状态（entries / scalar values / lists / usage
  / stats / nextSeq），供 commit 与各类 scan。
- ``InMemoryStorage``：实现 Storage Protocol 的异步外壳，转调 state。

seq 从 1 起、按 commit 顺序递增；message 条目计入 stats.messageCount；
value/list 用 ``namespace\u0000key`` 物理键。fork 属 A3，本模块不含。
"""

from __future__ import annotations

from dataclasses import replace
from typing import TypeVar, cast

from ...agent_core.types import Usage
from ..errors import SessionCorruptError
from ..types import Entry
from ..values import (
    ListElement,
    ListReadOptions,
    StoredValue,
    Value,
    ValueList,
    resolve_list_read_options,
)
from .commit import (
    CommittedEntryWrite,
    CommittedListWrite,
    CommittedUsageWrite,
    CommittedValueWrite,
    CommittedWrite,
    add_usage,
    prepare_storage_commit,
    validate_committed_writes,
)
from .storage_types import (
    BranchScan,
    CommitResult,
    Context,
    EntryScan,
    EntryStructure,
    SessionStats,
    UsageRow,
    UsageScan,
    Write,
)

T = TypeVar("T")


def _physical_key(namespace: str, key: str) -> str:
    return f"{namespace}\u0000{key}"


class InMemoryStorageState:
    """物化会话状态。数据库后端不应用这个类，而应查询索引化的持久状态。"""

    def __init__(self) -> None:
        self._entries: dict[str, Entry] = {}
        self._entries_by_seq: list[Entry] = []
        self._scalar_values: dict[str, StoredValue[object]] = {}
        self._list_values: dict[str, list[ListElement[object]]] = {}
        self._list_addresses: dict[str, ValueList[object]] = {}
        self._usage: dict[str, UsageRow] = {}
        self._stats = SessionStats(message_count=0, usage=Usage())
        self._next_seq = 1

    # -- commit --------------------------------------------------------------

    def prepare_and_apply(self, writes: list[Write], timestamp: int) -> CommitResult:
        prepared = prepare_storage_commit(writes, self._next_seq, timestamp)
        validate_committed_writes(
            prepared.writes,
            self._next_seq,
            _ValidationView(self._entries, self._usage),
        )
        stats = self._apply_validated(prepared.writes)
        return CommitResult(
            first_seq=prepared.first_seq,
            seqs=prepared.seqs,
            timestamp=prepared.timestamp,
            stats=stats,
        )

    def _apply_validated(self, writes: list[CommittedWrite]) -> SessionStats:
        for write in writes:
            if isinstance(write, CommittedEntryWrite):
                entry = write.entry
                self._entries[entry.id] = entry
                self._entries_by_seq.append(entry)
                if entry.type == "message":
                    self._stats = replace(
                        self._stats, message_count=self._stats.message_count + 1
                    )
                self._next_seq = entry.seq + 1
            elif isinstance(write, CommittedUsageWrite):
                row = write.row
                self._usage[row.id] = row
                self._stats = replace(
                    self._stats, usage=add_usage(self._stats.usage, row.usage)
                )
                self._next_seq = row.seq + 1
            elif isinstance(write, CommittedValueWrite):
                key = _physical_key(write.namespace, write.key)
                if write.op == "delete":
                    self._scalar_values.pop(key, None)
                else:
                    self._scalar_values[key] = StoredValue(
                        address=Value(namespace=write.namespace, key=write.key),
                        value=write.value,
                        seq=write.seq,
                    )
                self._next_seq = write.seq + 1
            elif isinstance(write, CommittedListWrite):
                key = _physical_key(write.namespace, write.key)
                if write.op == "delete":
                    self._list_values.pop(key, None)
                    self._list_addresses.pop(key, None)
                else:
                    self._list_addresses.setdefault(
                        key, ValueList(namespace=write.namespace, key=write.key)
                    )
                    self._list_values.setdefault(key, []).append(
                        ListElement(seq=write.seq, value=write.value)
                    )
                self._next_seq = write.seq + 1
            else:  # pragma: no cover - 联合已穷尽
                raise SessionCorruptError(f"unknown committed write: {write!r}")
        return self._stats

    # -- reads ---------------------------------------------------------------

    def get_entries(self, ids: list[str]) -> dict[str, Entry]:
        found: dict[str, Entry] = {}
        for id_ in ids:
            entry = self._entries.get(id_)
            if entry is not None:
                found[id_] = entry
        return found

    def get_value(self, address: Value[T]) -> StoredValue[T] | None:
        stored = self._scalar_values.get(_physical_key(address.namespace, address.key))
        return cast("StoredValue[T] | None", stored)

    def scan_values(self, prefix: Value[T]) -> list[StoredValue[T]]:
        matched = [
            stored
            for stored in self._scalar_values.values()
            if stored.address.namespace == prefix.namespace
            and stored.address.key.startswith(prefix.key)
        ]
        ordered = sorted(matched, key=lambda s: s.address.key)
        return cast("list[StoredValue[T]]", ordered)

    def read_list(
        self, address: ValueList[T], options: ListReadOptions | None
    ) -> list[ListElement[T]]:
        resolved = resolve_list_read_options(options)
        elements = self._list_values.get(
            _physical_key(address.namespace, address.key), []
        )
        if resolved.cursor is not None:
            cursor_seq = resolved.cursor.seq
            if resolved.order == "asc":
                elements = [e for e in elements if e.seq > cursor_seq]
            else:
                elements = [e for e in elements if e.seq < cursor_seq]
        ordered = elements if resolved.order == "asc" else list(reversed(elements))
        return cast("list[ListElement[T]]", ordered[: resolved.limit])

    def scan_branch(self, query: BranchScan) -> list[Entry]:
        start = self._entries.get(query.start)
        if start is None:
            raise SessionCorruptError(f"Unknown branch start: {query.start}")

        path: list[Entry] = []
        entry: Entry | None = start
        while entry is not None:
            path.append(entry)
            if entry.parent_id is None:
                break
            entry = self._entries.get(entry.parent_id)
            if entry is None:
                raise SessionCorruptError("Corrupt branch: missing parent")

        if query.order == "oldestFirst":
            path.reverse()

        stopped: list[Entry] = []
        for candidate in path:
            stopped.append(candidate)
            if candidate.id == query.stop_at_id or candidate.type == query.stop_at_type:
                break

        filtered = [
            c
            for c in stopped
            if (query.type is None or c.type == query.type)
            and (query.custom_type is None or c.custom_type == query.custom_type)
            and (
                query.cursor is None
                or (
                    c.seq > query.cursor.seq
                    if query.order == "oldestFirst"
                    else c.seq < query.cursor.seq
                )
            )
        ]
        return filtered if query.limit is None else filtered[: max(0, query.limit)]

    def scan_branch_structure(self, query: BranchScan) -> list[EntryStructure]:
        return [
            EntryStructure(
                id=e.id,
                parent_id=e.parent_id,
                seq=e.seq,
                timestamp=e.timestamp,
                type=e.type,
                custom_type=e.custom_type,
            )
            for e in self.scan_branch(query)
        ]

    def scan_entries(self, query: EntryScan) -> list[Entry]:
        limit = None if query.limit is None else max(0, query.limit)
        source = (
            reversed(self._entries_by_seq)
            if query.order == "desc"
            else self._entries_by_seq
        )
        out: list[Entry] = []
        for entry in source:
            if limit is not None and len(out) >= limit:
                break
            if (
                (query.type is None or entry.type == query.type)
                and (query.custom_type is None or entry.custom_type == query.custom_type)
                and (query.from_seq is None or entry.seq >= query.from_seq)
                and (query.to_seq is None or entry.seq <= query.to_seq)
            ):
                out.append(entry)
        return out

    def scan_usage(self, query: UsageScan) -> list[UsageRow]:
        rows = [
            row
            for row in self._usage.values()
            if (query.from_seq is None or row.seq >= query.from_seq)
            and (query.to_seq is None or row.seq <= query.to_seq)
        ]
        rows.sort(key=lambda r: r.seq, reverse=query.order == "desc")
        return rows if query.limit is None else rows[: max(0, query.limit)]

    def get_stats(self) -> SessionStats:
        return self._stats

    def get_next_seq(self) -> int:
        return self._next_seq


class _ValidationView:
    """给 validate_committed_writes 的只读视图（id 存在性判定）。"""

    def __init__(
        self, entries: dict[str, Entry], usage: dict[str, UsageRow]
    ) -> None:
        self._entries = entries
        self._usage = usage

    def has_entry_or_usage_id(self, id: str) -> bool:
        return id in self._entries or id in self._usage

    def has_entry_id(self, id: str) -> bool:
        return id in self._entries


class InMemoryStorage:
    """实现 Storage Protocol 的内存后端（异步外壳，转调 state；单进程/测试用）。"""

    def __init__(self, state: InMemoryStorageState | None = None) -> None:
        self._state = state if state is not None else InMemoryStorageState()
        self._clock = 0

    def _next_timestamp(self) -> int:
        self._clock += 1
        return self._clock

    async def commit(self, writes: list[Write], context: Context) -> CommitResult:
        return self._state.prepare_and_apply(writes, self._next_timestamp())

    async def get_entries(
        self, ids: list[str], context: Context
    ) -> dict[str, Entry]:
        return self._state.get_entries(ids)

    async def get_value(
        self, address: Value[T], context: Context
    ) -> StoredValue[T] | None:
        return self._state.get_value(address)

    async def scan_values(
        self, prefix: Value[T], context: Context
    ) -> list[StoredValue[T]]:
        return self._state.scan_values(prefix)

    async def read_list(
        self,
        address: ValueList[T],
        options: ListReadOptions | None,
        context: Context,
    ) -> list[ListElement[T]]:
        return self._state.read_list(address, options)

    async def scan_branch(
        self, query: BranchScan, context: Context
    ) -> list[Entry]:
        return self._state.scan_branch(query)

    async def scan_branch_structure(
        self, query: BranchScan, context: Context
    ) -> list[EntryStructure]:
        return self._state.scan_branch_structure(query)

    async def scan_entries(
        self, query: EntryScan, context: Context
    ) -> list[Entry]:
        return self._state.scan_entries(query)

    async def scan_usage(
        self, query: UsageScan, context: Context
    ) -> list[UsageRow]:
        return self._state.scan_usage(query)

    async def get_stats(self, context: Context) -> SessionStats:
        return self._state.get_stats()

    async def close(self, context: Context) -> None:
        return None


__all__ = ["InMemoryStorageState", "InMemoryStorage"]
