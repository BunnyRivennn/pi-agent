"""StorageBackedSession（对齐官方 session/session.ts + mutation-line.ts）。

- MutationLine：单会话读-改-写作业串行化（Python 用 asyncio.Lock + sealed 标记）。
- StorageBackedSession：写走 mutation 屏障；appendToBranch = 原子提交
  (insert_entry + set_value(branch_tip))；branch tip 存在 value 侧存储。
- 错误类对齐官方（SessionInvariantError 等）。

pending assistant 消息不可持久化（stop_reason == "pending"）。
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TypeVar

from ..agent_core.types import AgentMessage
from .errors import CompactionRaceError, SessionError
from .ids import new_entry_id
from .storage import (
    BranchScan,
    CommitResult,
    Context,
    EntryScan,
    SessionStats,
    Storage,
    Write,
    insert_entry,
)
from .types import CompactionEntry, CustomEntry, Entry, JsonValue, MessageEntry
from .values import (
    ListElement,
    ListReadOptions,
    StoredValue,
    Value,
    ValueList,
    branch_tip,
    entry_label,
    session_name,
)
from .values import (
    append_list as append_list_write,
)
from .values import (
    delete_list as delete_list_write,
)
from .values import (
    delete_value as delete_value_write,
)
from .values import (
    set_value as set_value_write,
)

T = TypeVar("T")

DEFAULT_BRANCH = "main"


@dataclass(slots=True)
class SessionMetadata:
    """会话身份元数据（由 SessionRepo 赋予；不进 entry 流）。"""

    id: str
    created_at: int = 0
    default_branch: str = DEFAULT_BRANCH
    extra: dict[str, JsonValue] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 错误类（对齐官方 session.ts:44-93）
# ---------------------------------------------------------------------------


class SessionInvariantError(SessionError):
    """持久状态内部不一致，无法安全推进。"""


class SessionInvalidBranchError(SessionError):
    def __init__(self, branch: str, reason: str) -> None:
        super().__init__(f"Invalid branch {branch!r}: {reason}")
        self.branch = branch
        self.reason = reason


class SessionBranchExistsError(SessionError):
    def __init__(self, branch: str) -> None:
        super().__init__(f"Branch already exists: {branch}")
        self.branch = branch


class SessionPendingAssistantMessageError(SessionError):
    def __init__(self) -> None:
        super().__init__("Cannot persist a pending assistant message")


class SessionUnknownTargetError(SessionError):
    def __init__(self, target_id: str) -> None:
        super().__init__(f"Unknown target: {target_id}")
        self.target_id = target_id


class SessionClosedError(SessionError):
    def __init__(self) -> None:
        super().__init__("Session is closed")


# ---------------------------------------------------------------------------
# MutationLine：单会话作业串行化
# ---------------------------------------------------------------------------


class MutationLine:
    """把单会话的完整读-改-写作业排队串行执行。

    官方用 Promise 尾链；Python 用 asyncio.Lock 达到同一效果——同一时刻只有一个
    mutation 作业持锁。seal 后拒绝新作业。
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._sealed_error: Exception | None = None

    async def run(self, operation: Callable[[], Awaitable[T]]) -> T:
        if self._sealed_error is not None:
            raise self._sealed_error
        async with self._lock:
            if self._sealed_error is not None:
                raise self._sealed_error
            return await operation()

    async def seal(self, error: Exception) -> None:
        if self._sealed_error is None:
            self._sealed_error = error
        # 等待在途作业排空
        async with self._lock:
            return None


# 注：官方会拒绝持久化 pending assistant 消息（stopReason=="pending"）。
# 我们的 StopReason 模型无 "pending" 状态，故该守卫当前不可达；保留
# SessionPendingAssistantMessageError 类仅为 API 对齐，待模型引入 pending 再启用。


# ---------------------------------------------------------------------------
# SessionMutation：mutation 作业内的受控写句柄
# ---------------------------------------------------------------------------


class SessionMutation:
    """一次 mutation 作业内暴露给回调的句柄：至多 commit 一次。

    在 mutation 回调之外使用会抛错（active=False）。
    """

    def __init__(self, storage: Storage) -> None:
        self._storage = storage
        self._active = True
        self._committed = False

    async def commit(self, writes: list[Write], context: Context) -> CommitResult:
        self._assert_active()
        if self._committed:
            raise SessionError("SessionMutation commit already attempted")
        self._committed = True
        return await self._storage.commit(writes, context)

    async def get_entries(self, ids: list[str], context: Context) -> dict[str, Entry]:
        self._assert_active()
        return await self._storage.get_entries(ids, context)

    async def get_value(
        self, address: Value[T], context: Context
    ) -> StoredValue[T] | None:
        self._assert_active()
        return await self._storage.get_value(address, context)

    def _deactivate(self) -> None:
        self._active = False

    def _assert_active(self) -> None:
        if not self._active:
            raise SessionError(
                "SessionMutation cannot be used outside its mutation callback"
            )


# ---------------------------------------------------------------------------
# Branch：分支视图
# ---------------------------------------------------------------------------


class StorageBackedBranch:
    def __init__(self, name: str, session: StorageBackedSession) -> None:
        self.name = name
        self._session = session

    async def get_tip_id(self, context: Context) -> str | None:
        return await self._session.get_branch_tip(self.name, context)

    async def find_entries(
        self, query: BranchScan | None, context: Context
    ) -> list[Entry]:
        start = query.start if query is not None else None
        if start is None or start == "":
            tip = await self.get_tip_id(context)
            if tip is None:
                return []
            start = tip
        resolved = BranchScan(
            start=start,
            stop_at_type=query.stop_at_type if query else None,
            stop_at_id=query.stop_at_id if query else None,
            type=query.type if query else None,
            custom_type=query.custom_type if query else None,
            order=query.order if query else "newestFirst",
            limit=query.limit if query else None,
            cursor=query.cursor if query else None,
        )
        return await self._session.scan_branch(resolved, context)

    async def find_entry(
        self, query: BranchScan | None, context: Context
    ) -> Entry | None:
        entries = await self.find_entries(query, context)
        return entries[0] if entries else None

    async def append_message(self, message: AgentMessage, context: Context) -> str:
        return await self._session.append_to_branch(
            self.name, ("message", message, None, None), context
        )

    async def append_custom_entry(
        self, custom_type: str, data: JsonValue, context: Context
    ) -> str:
        return await self._session.append_to_branch(
            self.name, ("custom", None, custom_type, data), context
        )


# ("message", message, None, None) | ("custom", None, custom_type, data)
_AppendSpec = tuple[str, "AgentMessage | None", "str | None", JsonValue]


# ---------------------------------------------------------------------------
# StorageBackedSession
# ---------------------------------------------------------------------------


class StorageBackedSession:
    """会话对象：读直连 storage，写走 mutation 屏障串行化。"""

    def __init__(
        self,
        metadata: SessionMetadata,
        storage: Storage,
        mutation_line: MutationLine | None = None,
        on_close: Callable[[], None] | None = None,
    ) -> None:
        self.metadata = metadata
        self._storage = storage
        self._mutation_line = mutation_line or MutationLine()
        self._on_close = on_close
        self._branches: dict[str, StorageBackedBranch] = {}
        self._state = "open"
        self._close_task: asyncio.Task[None] | None = None

    # -- mutation ------------------------------------------------------------

    async def mutate(
        self,
        callback: Callable[[SessionMutation, Context], Awaitable[T]],
        context: Context,
    ) -> T:
        self._assert_open()

        async def job() -> T:
            mutator = SessionMutation(self._storage)
            try:
                return await callback(mutator, context)
            finally:
                mutator._deactivate()

        return await self._mutation_line.run(job)

    # -- reads (直连 storage) -------------------------------------------------

    async def get_entries(self, ids: list[str], context: Context) -> dict[str, Entry]:
        self._assert_open()
        return await self._storage.get_entries(ids, context)

    async def get_entry(self, id: str, context: Context) -> Entry | None:
        return (await self.get_entries([id], context)).get(id)

    async def get_value(
        self, address: Value[T], context: Context
    ) -> StoredValue[T] | None:
        self._assert_open()
        return await self._storage.get_value(address, context)

    async def scan_values(
        self, prefix: Value[T], context: Context
    ) -> list[StoredValue[T]]:
        self._assert_open()
        return await self._storage.scan_values(prefix, context)

    async def read_list(
        self, address: ValueList[T], options: ListReadOptions | None, context: Context
    ) -> list[ListElement[T]]:
        self._assert_open()
        return await self._storage.read_list(address, options, context)

    async def scan_branch(self, query: BranchScan, context: Context) -> list[Entry]:
        self._assert_open()
        return await self._storage.scan_branch(query, context)

    async def get_stats(self, context: Context) -> SessionStats:
        self._assert_open()
        return await self._storage.get_stats(context)

    async def find_entries(
        self, query: EntryScan | None, context: Context
    ) -> list[Entry]:
        self._assert_open()
        return await self._storage.scan_entries(query or EntryScan(order="desc"), context)

    async def find_entry(
        self, query: EntryScan | None, context: Context
    ) -> Entry | None:
        q = query or EntryScan(order="desc")
        q.limit = 1 if q.limit is None else min(q.limit, 1)
        entries = await self.find_entries(q, context)
        return entries[0] if entries else None

    async def get_name(self, context: Context) -> str | None:
        stored = await self.get_value(session_name(), context)
        return stored.value if stored is not None else None

    async def get_label(self, target_id: str, context: Context) -> str | None:
        stored = await self.get_value(entry_label(target_id), context)
        return stored.value if stored is not None else None

    # -- branches ------------------------------------------------------------

    async def branch(self, name: str, context: Context) -> StorageBackedBranch | None:
        self._assert_valid_branch_name(name)
        if (await self.get_value(branch_tip(name), context)) is None:
            return None
        return self._get_or_create_branch(name)

    async def create_branch(
        self, name: str, at: str | None, context: Context
    ) -> StorageBackedBranch:
        self._assert_open()
        self._assert_valid_branch_name(name)

        async def job(mutator: SessionMutation, ctx: Context) -> None:
            if (await mutator.get_value(branch_tip(name), ctx)) is not None:
                raise SessionBranchExistsError(name)
            if at is not None and at not in (await mutator.get_entries([at], ctx)):
                raise SessionUnknownTargetError(at)
            await mutator.commit([set_value_write(branch_tip(name), at)], ctx)

        await self.mutate(job, context)
        return self._get_or_create_branch(name)

    async def get_branch_tip(self, name: str, context: Context) -> str | None:
        stored = await self.get_value(branch_tip(name), context)
        if stored is None:
            raise SessionInvariantError(f"Unknown branch: {name}")
        return stored.value

    async def append_to_branch(
        self, name: str, spec: _AppendSpec, context: Context
    ) -> str:
        self._assert_open()
        kind, message, custom_type, data = spec
        entry_id = new_entry_id(self._branches)  # 局部唯一即可；存储层再全局校验

        async def job(mutator: SessionMutation, ctx: Context) -> None:
            tip = await mutator.get_value(branch_tip(name), ctx)
            if tip is None:
                raise SessionInvariantError(f"Unknown branch: {name}")
            entry: Entry
            if kind == "message":
                entry = MessageEntry(
                    id=entry_id, parent_id=tip.value, message=message
                )
            else:
                entry = CustomEntry(
                    id=entry_id,
                    parent_id=tip.value,
                    custom_type=custom_type,
                    data=data,
                )
            await mutator.commit(
                [insert_entry(entry), set_value_write(branch_tip(name), entry_id)],
                ctx,
            )

        await self.mutate(job, context)
        return entry_id

    async def append_compaction_to_branch(
        self,
        name: str,
        *,
        summary: str,
        retained_tail: list[AgentMessage],
        tokens_before: int,
        expected_tip: str,
        usage: dict[str, object] | None = None,
        from_hook: bool = False,
        context: Context,
    ) -> str:
        """乐观地把一个 CompactionEntry 接到 branch 上。

        ``expected_tip`` 是调用方算切点/生成摘要时看到的 tip；提交前在屏障内校验
        tip 未变，若已被新消息推进则抛 ``CompactionRaceError``（调用方静默跳过、
        下轮重试）。原始 entry 从不删除——压缩只是 append 摘要 + 推进 tip，
        由 build_context 在读取时从摘要处截断。
        """
        self._assert_open()
        entry_id = new_entry_id(self._branches)

        async def job(mutator: SessionMutation, ctx: Context) -> None:
            tip = await mutator.get_value(branch_tip(name), ctx)
            if tip is None:
                raise SessionInvariantError(f"Unknown branch: {name}")
            if tip.value != expected_tip:
                raise CompactionRaceError(
                    f"branch {name!r} tip moved during compaction: "
                    f"expected {expected_tip}, found {tip.value}"
                )
            entry = CompactionEntry(
                id=entry_id,
                parent_id=tip.value,
                summary=summary,
                retained_tail=retained_tail,
                tokens_before=tokens_before,
                usage=usage,
                from_hook=from_hook,
            )
            await mutator.commit(
                [insert_entry(entry), set_value_write(branch_tip(name), entry_id)],
                ctx,
            )

        await self.mutate(job, context)
        return entry_id

    async def set_value(self, address: Value[T], next_value: T, context: Context) -> None:
        async def job(mutator: SessionMutation, ctx: Context) -> None:
            await mutator.commit([set_value_write(address, next_value)], ctx)

        await self.mutate(job, context)

    async def delete_value(self, address: Value[T], context: Context) -> None:
        async def job(mutator: SessionMutation, ctx: Context) -> None:
            await mutator.commit([delete_value_write(address)], ctx)

        await self.mutate(job, context)

    async def append_list(
        self, address: ValueList[T], element: T, context: Context
    ) -> None:
        async def job(mutator: SessionMutation, ctx: Context) -> None:
            await mutator.commit([append_list_write(address, element)], ctx)

        await self.mutate(job, context)

    async def delete_list(self, address: ValueList[T], context: Context) -> None:
        async def job(mutator: SessionMutation, ctx: Context) -> None:
            await mutator.commit([delete_list_write(address)], ctx)

        await self.mutate(job, context)

    async def set_name(self, name: str | None, context: Context) -> None:
        if name is None:
            await self.delete_value(session_name(), context)
        else:
            await self.set_value(session_name(), name, context)

    async def set_label(
        self, target_id: str, label: str | None, context: Context
    ) -> None:
        address = entry_label(target_id)
        if label is None:
            await self.delete_value(address, context)
        else:
            await self.set_value(address, label, context)

    # -- lifecycle -----------------------------------------------------------

    async def close(self, context: Context) -> None:
        if self._state == "closed":
            return
        self._state = "closing"
        await self._mutation_line.seal(SessionClosedError())
        await self._storage.close(context)
        self._state = "closed"
        if self._on_close is not None:
            self._on_close()

    # -- internals -----------------------------------------------------------

    def _get_or_create_branch(self, name: str) -> StorageBackedBranch:
        branch = self._branches.get(name)
        if branch is None:
            branch = StorageBackedBranch(name, self)
            self._branches[name] = branch
        return branch

    def _assert_valid_branch_name(self, name: str) -> None:
        if len(name) == 0:
            raise SessionInvalidBranchError(name, "branch name must not be empty")
        if "\u0000" in name:
            raise SessionInvalidBranchError(name, "branch name must not contain \\u0000")

    def _assert_open(self) -> None:
        if self._state != "open":
            raise SessionClosedError()


__all__ = [
    "MutationLine",
    "SessionMutation",
    "StorageBackedBranch",
    "StorageBackedSession",
    "SessionInvariantError",
    "SessionInvalidBranchError",
    "SessionBranchExistsError",
    "SessionPendingAssistantMessageError",
    "SessionUnknownTargetError",
    "SessionClosedError",
]
