"""SessionRepo（对齐官方 session/*repo）：多会话生命周期管理。

职责边界（与 Storage 分层）：
- Storage：单会话内的 entry/value/list/usage 读写。
- SessionRepo：跨会话的 create / open / list / delete / fork，以及决定
  "每个会话的存储落在哪里"（内存字典 or 同一个 .db 文件里的一行）。

两个实现：
- InMemorySessionRepo：进程内字典，测试/临时用。
- SqliteSessionRepo：**单个 db 文件装全部会话**，每张表用 session_id 隔离。

新建会话会自动建默认分支（default_branch，tip 指向 None = 空分支）。
"""

from __future__ import annotations

import os
import time
from typing import Protocol

from .ids import assert_valid_session_id, new_session_id
from .session import (
    DEFAULT_BRANCH,
    SessionMetadata,
    SessionMutation,
    StorageBackedSession,
)
from .storage import Context, InMemoryStorage, Storage
from .storage.sqlite import SqliteDatabase, SqliteStorage
from .values import branch_tip, set_value


async def _ensure_default_branch(
    session: StorageBackedSession, branch: str, context: Context
) -> None:
    # 幂等：仅当 tip 不存在时建空分支（tip=None）。
    if (await session.get_value(branch_tip(branch), context)) is not None:
        return

    async def job(mutator: SessionMutation, ctx: Context) -> None:
        await mutator.commit([set_value(branch_tip(branch), None)], ctx)

    await session.mutate(job, context)


class SessionRepo(Protocol):
    """会话仓库接口。"""

    async def create(
        self, context: Context, session_id: str | None = ...
    ) -> StorageBackedSession: ...

    async def open(
        self, session_id: str, context: Context
    ) -> StorageBackedSession | None: ...

    async def list_ids(self, context: Context) -> list[str]: ...

    async def delete(self, session_id: str, context: Context) -> bool: ...

    async def close(self, context: Context) -> None: ...


# ---------------------------------------------------------------------------
# InMemorySessionRepo
# ---------------------------------------------------------------------------


class InMemorySessionRepo:
    """进程内会话仓库：每会话一个 InMemoryStorage，存活于本 repo 实例。"""

    def __init__(self, default_branch: str = DEFAULT_BRANCH) -> None:
        self._default_branch = default_branch
        self._storages: dict[str, InMemoryStorage] = {}
        self._metadata: dict[str, SessionMetadata] = {}

    async def create(
        self, context: Context, session_id: str | None = None
    ) -> StorageBackedSession:
        sid = session_id or new_session_id()
        assert_valid_session_id(sid)
        if sid in self._storages:
            raise ValueError(f"Session already exists: {sid}")
        storage = InMemoryStorage()
        self._storages[sid] = storage
        meta = SessionMetadata(
            id=sid,
            created_at=int(time.time() * 1000),
            default_branch=self._default_branch,
        )
        self._metadata[sid] = meta
        session = StorageBackedSession(meta, storage)
        await _ensure_default_branch(session, self._default_branch, context)
        return session

    async def open(
        self, session_id: str, context: Context
    ) -> StorageBackedSession | None:
        storage = self._storages.get(session_id)
        if storage is None:
            return None
        return StorageBackedSession(self._metadata[session_id], storage)

    async def list_ids(self, context: Context) -> list[str]:
        return sorted(self._storages)

    async def delete(self, session_id: str, context: Context) -> bool:
        existed = session_id in self._storages
        self._storages.pop(session_id, None)
        self._metadata.pop(session_id, None)
        return existed

    async def close(self, context: Context) -> None:
        for storage in self._storages.values():
            await storage.close(context)


# ---------------------------------------------------------------------------
# SqliteSessionRepo
# ---------------------------------------------------------------------------


class SqliteSessionRepo:
    """落盘会话仓库：**单个 db 文件装全部会话**。

    这里就是 "db 文件最终存哪" 的落地点：由 ``path`` 指定的单一文件，
    会话之间靠 ``sessions`` 表 + 各表的 session_id 列隔离。

    对比 jsonl 后端的 "一会话一文件"：那是 append-only 文本流的固有约束；
    关系库没有这个限制，list/排序/跨会话查询都能走 SQL。
    """

    def __init__(self, path: str, default_branch: str = DEFAULT_BRANCH) -> None:
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._path = path
        self._default_branch = default_branch
        self._db = SqliteDatabase(path)

    @property
    def path(self) -> str:
        """db 文件路径（全部会话都在这一个文件里）。"""
        return self._path

    async def create(
        self, context: Context, session_id: str | None = None
    ) -> StorageBackedSession:
        sid = session_id or new_session_id()
        assert_valid_session_id(sid)
        if self._db.has_session(sid):
            raise ValueError(f"Session already exists: {sid}")
        created_at = int(time.time() * 1000)
        storage: Storage = SqliteStorage.attach(self._db, sid)
        meta = SessionMetadata(
            id=sid,
            created_at=created_at,
            default_branch=self._default_branch,
        )
        session = StorageBackedSession(meta, storage)
        await _ensure_default_branch(session, self._default_branch, context)
        return session

    async def open(
        self, session_id: str, context: Context
    ) -> StorageBackedSession | None:
        if not self._db.has_session(session_id):
            return None
        storage: Storage = SqliteStorage.attach(self._db, session_id)
        meta = SessionMetadata(id=session_id, default_branch=self._default_branch)
        return StorageBackedSession(meta, storage)

    async def list_ids(self, context: Context) -> list[str]:
        """返回全部会话 id，**按创建时间倒序**（最新在前）。"""
        return [sid for sid, _ in self._db.list_sessions()]

    async def list_sessions(self, context: Context) -> list[tuple[str, int]]:
        """返回 [(session_id, created_at)]，按创建时间倒序。"""
        return self._db.list_sessions()

    async def delete(self, session_id: str, context: Context) -> bool:
        return self._db.delete_session(session_id)

    async def close(self, context: Context) -> None:
        self._db.close()


__all__ = [
    "SessionRepo",
    "InMemorySessionRepo",
    "SqliteSessionRepo",
]
