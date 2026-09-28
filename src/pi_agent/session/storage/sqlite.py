"""SQLite 存储后端（实现 Storage Protocol；对齐 InMemoryStorage 语义）。

设计要点（区别于 InMemoryStorageState 的内存物化）：
- 状态留在磁盘：每个 Storage 方法 = 一条 SQL，不把整会话读进内存。
- 聚合行 ``sessions``（每会话一行）在 commit 的同一事务内 UPDATE，getStats 直接读一行（O(1)）。
- 复用 commit.py 的纯函数管线（prepare_storage_commit / validate_committed_writes），
  仅把校验视图换成 SQL EXISTS。
- **单库多会话**：一个 db 文件装任意多个会话，每张表都用 session_id 作用域隔离
  （对齐官方 session-backends/sqlite-node 的 schema）。表结构与 PG 同构，换后端只改 SQL 方言。

payload 编码：entry 用 serialize.entry_to_dict → JSON；value/list/usage 值用 JSON。
"""

from __future__ import annotations

import json
import sqlite3
import time
from typing import TypeVar, cast

from ...agent_core.types import Usage
from ..errors import SessionCorruptError
from ..serialize import (
    entry_from_dict,
    entry_to_dict,
    usage_from_dict,
    usage_to_dict,
)
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

STORAGE_VERSION = 1

_DDL = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    id            TEXT PRIMARY KEY,
    created_at    INTEGER NOT NULL,
    metadata      TEXT,
    message_count INTEGER NOT NULL DEFAULT 0,
    usage         TEXT NOT NULL,
    next_seq      INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS entries (
    session_id TEXT NOT NULL,
    id         TEXT NOT NULL,
    parent_id  TEXT,
    seq        INTEGER NOT NULL,
    timestamp  INTEGER NOT NULL,
    type       TEXT NOT NULL,
    custom_type TEXT,
    payload    TEXT NOT NULL,
    PRIMARY KEY (session_id, id)
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_entries_seq ON entries(session_id, seq);
CREATE INDEX IF NOT EXISTS idx_entries_parent ON entries(session_id, parent_id);
CREATE INDEX IF NOT EXISTS idx_entries_type ON entries(session_id, type);

CREATE TABLE IF NOT EXISTS values_kv (
    session_id TEXT NOT NULL,
    namespace TEXT NOT NULL,
    key       TEXT NOT NULL,
    seq       INTEGER NOT NULL,
    value     TEXT NOT NULL,
    PRIMARY KEY (session_id, namespace, key)
);

CREATE TABLE IF NOT EXISTS lists_kv (
    session_id TEXT NOT NULL,
    namespace TEXT NOT NULL,
    key       TEXT NOT NULL,
    seq       INTEGER NOT NULL,
    value     TEXT NOT NULL,
    PRIMARY KEY (session_id, namespace, key, seq)
);

CREATE TABLE IF NOT EXISTS usage (
    session_id TEXT NOT NULL,
    id         TEXT NOT NULL,
    seq        INTEGER NOT NULL,
    usage      TEXT NOT NULL,
    adjustment INTEGER NOT NULL DEFAULT 0,
    entry_id   TEXT,
    details    TEXT,
    PRIMARY KEY (session_id, id)
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_usage_seq ON usage(session_id, seq);
"""


def _empty_usage_json() -> str:
    return json.dumps(usage_to_dict(Usage()))


class SqliteDatabase:
    """一个 db 文件 = 一个会话容器。

    持有唯一的 sqlite 连接，多个 :class:`SqliteStorage`（每会话一个）共享它。
    连接层面的 PRAGMA 与建表只做一次。
    """

    def __init__(self, path: str = ":memory:") -> None:
        # isolation_level=None：关掉 sqlite3 的隐式事务管理，改由 SqliteStorage 用显式
        # BEGIN IMMEDIATE / commit / rollback 精确控制，配合 seq 计数器避免竞态。
        self._conn = sqlite3.connect(path, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._init_schema()

    @property
    def conn(self) -> sqlite3.Connection:
        return self._conn

    def _init_schema(self) -> None:
        # autocommit 模式下 executescript 自成事务；seed 行用显式事务。
        self._conn.executescript(_DDL)
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            row = self._conn.execute(
                "SELECT value FROM meta WHERE key='storage_version'"
            ).fetchone()
            if row is None:
                self._conn.execute(
                    "INSERT INTO meta(key, value) VALUES('storage_version', ?)",
                    (str(STORAGE_VERSION),),
                )
        except BaseException:
            self._conn.rollback()
            raise
        self._conn.commit()

    def has_session(self, session_id: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM sessions WHERE id=? LIMIT 1", (session_id,)
        ).fetchone()
        return row is not None

    def create_session(self, session_id: str, created_at: int) -> None:
        """登记一个新会话行；已存在则抛错。"""
        try:
            self._conn.execute(
                "INSERT INTO sessions(id, created_at, metadata, message_count, usage, "
                "next_seq) VALUES(?,?,NULL,0,?,1)",
                (session_id, created_at, _empty_usage_json()),
            )
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"Session already exists: {session_id}") from exc

    def list_sessions(self) -> list[tuple[str, int]]:
        """返回 [(session_id, created_at)]，按创建时间倒序（最新在前）。"""
        rows = self._conn.execute(
            "SELECT id, created_at FROM sessions ORDER BY created_at DESC, id DESC"
        ).fetchall()
        return [(row["id"], int(row["created_at"])) for row in rows]

    def delete_session(self, session_id: str) -> bool:
        """删除一个会话及其全部数据（单事务）。"""
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            cur = self._conn.execute("DELETE FROM sessions WHERE id=?", (session_id,))
            existed = cur.rowcount > 0
            for table in ("entries", "values_kv", "lists_kv", "usage"):
                self._conn.execute(
                    f"DELETE FROM {table} WHERE session_id=?", (session_id,)  # noqa: S608
                )
        except BaseException:
            self._conn.rollback()
            raise
        self._conn.commit()
        return existed

    def close(self) -> None:
        self._conn.close()


class SqliteStorage:
    """实现 Storage Protocol 的 SQLite 后端（单会话视图）。

    每个实例代表**一个库文件里的一个会话**，所有读写都按 ``session_id`` 作用域。
    ``path=":memory:"`` 走进程内库（测试用）；否则落盘文件。

    两种构造方式：
    - ``SqliteStorage(path, session_id=...)``：自建库连接（会话不存在则自动登记）。
    - ``SqliteStorage.attach(db, session_id)``：复用已有的 :class:`SqliteDatabase`。
    """

    def __init__(
        self,
        path: str = ":memory:",
        session_id: str = "default",
        *,
        _db: SqliteDatabase | None = None,
        _owns_db: bool = True,
    ) -> None:
        self._db = _db if _db is not None else SqliteDatabase(path)
        self._owns_db = _owns_db
        self._session_id = session_id
        self._conn = self._db.conn
        if not self._db.has_session(session_id):
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                if not self._db.has_session(session_id):
                    self._db.create_session(session_id, int(time.time() * 1000))
            except BaseException:
                self._conn.rollback()
                raise
            self._conn.commit()

    @classmethod
    def attach(cls, db: SqliteDatabase, session_id: str) -> SqliteStorage:
        """在已打开的库上挂一个会话视图（不拥有该连接，close 不关库）。"""
        return cls(session_id=session_id, _db=db, _owns_db=False)

    @property
    def session_id(self) -> str:
        return self._session_id

    def _read_next_seq(self) -> int:
        """读持久 seq 计数器（sessions.next_seq）。

        必须在 commit 事务内调用：读-用-递增三步同处一个事务，才能避免
        并发写者拿到同一个 seq。PG 后端把这里换成 SEQUENCE/nextval 即可，
        竞态处理收敛在这一个方法 + _bump_next_seq 里。
        """
        row = self._conn.execute(
            "SELECT next_seq FROM sessions WHERE id=?", (self._session_id,)
        ).fetchone()
        if row is None:  # pragma: no cover - 会话行在 __init__ 建好
            raise SessionCorruptError(f"missing sessions row: {self._session_id}")
        return int(row["next_seq"])

    def _bump_next_seq(self, next_seq: int) -> None:
        self._conn.execute(
            "UPDATE sessions SET next_seq=? WHERE id=?", (next_seq, self._session_id)
        )

    # -- commit --------------------------------------------------------------

    async def commit(self, writes: list[Write], context: Context) -> CommitResult:
        timestamp = int(time.time() * 1000)
        # 单事务：分配 seq + 校验 + 写入行 + 更新聚合行，任一失败整体回滚。
        # BEGIN IMMEDIATE 立刻取写锁，确保 seq 的读-递增不被其他写者插队。
        try:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                first_seq = self._read_next_seq()
                prepared = prepare_storage_commit(writes, first_seq, timestamp)
                validate_committed_writes(
                    prepared.writes,
                    first_seq,
                    _SqliteValidationView(self._conn, self._session_id),
                )
                msg_delta = 0
                usage_delta: list[str] = []
                for w in prepared.writes:
                    if isinstance(w, CommittedEntryWrite):
                        self._insert_entry(w.entry)
                        if w.entry.type == "message":
                            msg_delta += 1
                    elif isinstance(w, CommittedUsageWrite):
                        self._insert_usage(w.row)
                        usage_delta.append(json.dumps(usage_to_dict(w.row.usage)))
                    elif isinstance(w, CommittedValueWrite):
                        self._apply_value(w)
                    elif isinstance(w, CommittedListWrite):
                        self._apply_list(w)
                    else:  # pragma: no cover
                        raise SessionCorruptError(f"unknown committed write: {w!r}")
                self._bump_next_seq(first_seq + len(prepared.writes))
                stats = self._update_stats(msg_delta, usage_delta)
            except BaseException:
                self._conn.rollback()
                raise
            self._conn.commit()
        except sqlite3.IntegrityError as exc:
            raise SessionCorruptError(f"commit integrity error: {exc}") from exc
        return CommitResult(
            first_seq=prepared.first_seq,
            seqs=prepared.seqs,
            timestamp=prepared.timestamp,
            stats=stats,
        )

    def _insert_entry(self, entry: Entry) -> None:
        self._conn.execute(
            "INSERT INTO entries(session_id, id, parent_id, seq, timestamp, type, "
            "custom_type, payload) VALUES(?,?,?,?,?,?,?,?)",
            (
                self._session_id,
                entry.id,
                entry.parent_id,
                entry.seq,
                entry.timestamp,
                entry.type,
                entry.custom_type,
                json.dumps(entry_to_dict(entry)),
            ),
        )

    def _insert_usage(self, row: UsageRow) -> None:
        self._conn.execute(
            "INSERT INTO usage(session_id, id, seq, usage, adjustment, entry_id, details) "
            "VALUES(?,?,?,?,?,?,?)",
            (
                self._session_id,
                row.id,
                row.seq,
                json.dumps(usage_to_dict(row.usage)),
                1 if row.adjustment else 0,
                row.entry_id,
                None if row.details is None else json.dumps(row.details),
            ),
        )

    def _apply_value(self, w: CommittedValueWrite) -> None:
        if w.op == "delete":
            self._conn.execute(
                "DELETE FROM values_kv WHERE session_id=? AND namespace=? AND key=?",
                (self._session_id, w.namespace, w.key),
            )
        else:
            self._conn.execute(
                "INSERT INTO values_kv(session_id, namespace, key, seq, value) "
                "VALUES(?,?,?,?,?) "
                "ON CONFLICT(session_id, namespace, key) DO UPDATE SET seq=excluded.seq, "
                "value=excluded.value",
                (self._session_id, w.namespace, w.key, w.seq, json.dumps(w.value)),
            )

    def _apply_list(self, w: CommittedListWrite) -> None:
        if w.op == "delete":
            self._conn.execute(
                "DELETE FROM lists_kv WHERE session_id=? AND namespace=? AND key=?",
                (self._session_id, w.namespace, w.key),
            )
        else:
            self._conn.execute(
                "INSERT INTO lists_kv(session_id, namespace, key, seq, value) "
                "VALUES(?,?,?,?,?)",
                (self._session_id, w.namespace, w.key, w.seq, json.dumps(w.value)),
            )

    def _update_stats(self, msg_delta: int, usage_json_list: list[str]) -> SessionStats:
        row = self._conn.execute(
            "SELECT message_count, usage FROM sessions WHERE id=?", (self._session_id,)
        ).fetchone()
        usage = usage_from_dict(json.loads(row["usage"]))
        for uj in usage_json_list:
            usage = add_usage(usage, usage_from_dict(json.loads(uj)))
        message_count = int(row["message_count"]) + msg_delta
        self._conn.execute(
            "UPDATE sessions SET message_count=?, usage=? WHERE id=?",
            (message_count, json.dumps(usage_to_dict(usage)), self._session_id),
        )
        return SessionStats(message_count=message_count, usage=usage)

    # -- reads ---------------------------------------------------------------

    def _row_to_entry(self, row: sqlite3.Row) -> Entry:
        return entry_from_dict(json.loads(row["payload"]))

    async def get_entries(
        self, ids: list[str], context: Context
    ) -> dict[str, Entry]:
        if not ids:
            return {}
        placeholders = ",".join("?" * len(ids))
        rows = self._conn.execute(
            f"SELECT id, payload FROM entries WHERE session_id=? AND id IN ({placeholders})",  # noqa: S608
            [self._session_id, *ids],
        ).fetchall()
        return {row["id"]: self._row_to_entry(row) for row in rows}

    async def get_value(
        self, address: Value[T], context: Context
    ) -> StoredValue[T] | None:
        row = self._conn.execute(
            "SELECT seq, value FROM values_kv WHERE session_id=? AND namespace=? AND key=?",
            (self._session_id, address.namespace, address.key),
        ).fetchone()
        if row is None:
            return None
        return StoredValue(
            address=address, value=cast("T", json.loads(row["value"])), seq=row["seq"]
        )

    async def scan_values(
        self, prefix: Value[T], context: Context
    ) -> list[StoredValue[T]]:
        rows = self._conn.execute(
            "SELECT namespace, key, seq, value FROM values_kv "
            "WHERE session_id=? AND namespace=? AND key LIKE ? ESCAPE '\\' ORDER BY key ASC",
            (self._session_id, prefix.namespace, _like_prefix(prefix.key)),
        ).fetchall()
        return [
            StoredValue(
                address=Value(namespace=row["namespace"], key=row["key"]),
                value=cast("T", json.loads(row["value"])),
                seq=row["seq"],
            )
            for row in rows
        ]

    async def read_list(
        self,
        address: ValueList[T],
        options: ListReadOptions | None,
        context: Context,
    ) -> list[ListElement[T]]:
        resolved = resolve_list_read_options(options)
        sql = "SELECT seq, value FROM lists_kv WHERE session_id=? AND namespace=? AND key=?"
        params: list[object] = [self._session_id, address.namespace, address.key]
        if resolved.cursor is not None:
            sql += " AND seq > ?" if resolved.order == "asc" else " AND seq < ?"
            params.append(resolved.cursor.seq)
        sql += " ORDER BY seq " + ("ASC" if resolved.order == "asc" else "DESC")
        sql += " LIMIT ?"
        params.append(resolved.limit)
        rows = self._conn.execute(sql, params).fetchall()
        return [
            ListElement(seq=row["seq"], value=cast("T", json.loads(row["value"])))
            for row in rows
        ]

    async def scan_branch(self, query: BranchScan, context: Context) -> list[Entry]:
        # 沿 parent_id 迭代上溯（起点必须存在）。
        start = self._conn.execute(
            "SELECT payload, parent_id FROM entries WHERE session_id=? AND id=?",
            (self._session_id, query.start),
        ).fetchone()
        if start is None:
            raise SessionCorruptError(f"Unknown branch start: {query.start}")

        path: list[Entry] = []
        cursor_row: sqlite3.Row | None = start
        while cursor_row is not None:
            entry = entry_from_dict(json.loads(cursor_row["payload"]))
            path.append(entry)
            parent_id = cursor_row["parent_id"]
            if parent_id is None:
                break
            cursor_row = self._conn.execute(
                "SELECT payload, parent_id FROM entries WHERE session_id=? AND id=?",
                (self._session_id, parent_id),
            ).fetchone()
            if cursor_row is None:
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

    async def scan_branch_structure(
        self, query: BranchScan, context: Context
    ) -> list[EntryStructure]:
        return [
            EntryStructure(
                id=e.id,
                parent_id=e.parent_id,
                seq=e.seq,
                timestamp=e.timestamp,
                type=e.type,
                custom_type=e.custom_type,
            )
            for e in await self.scan_branch(query, context)
        ]

    async def scan_entries(self, query: EntryScan, context: Context) -> list[Entry]:
        sql = "SELECT payload FROM entries"
        clauses: list[str] = ["session_id=?"]
        params: list[object] = [self._session_id]
        if query.type is not None:
            clauses.append("type=?")
            params.append(query.type)
        if query.custom_type is not None:
            clauses.append("custom_type=?")
            params.append(query.custom_type)
        if query.from_seq is not None:
            clauses.append("seq>=?")
            params.append(query.from_seq)
        if query.to_seq is not None:
            clauses.append("seq<=?")
            params.append(query.to_seq)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY seq " + ("DESC" if query.order == "desc" else "ASC")
        if query.limit is not None:
            sql += " LIMIT ?"
            params.append(max(0, query.limit))
        rows = self._conn.execute(sql, params).fetchall()
        return [entry_from_dict(json.loads(row["payload"])) for row in rows]

    async def scan_usage(self, query: UsageScan, context: Context) -> list[UsageRow]:
        sql = "SELECT id, seq, usage, adjustment, entry_id, details FROM usage"
        clauses: list[str] = ["session_id=?"]
        params: list[object] = [self._session_id]
        if query.from_seq is not None:
            clauses.append("seq>=?")
            params.append(query.from_seq)
        if query.to_seq is not None:
            clauses.append("seq<=?")
            params.append(query.to_seq)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY seq " + ("DESC" if query.order == "desc" else "ASC")
        if query.limit is not None:
            sql += " LIMIT ?"
            params.append(max(0, query.limit))
        rows = self._conn.execute(sql, params).fetchall()
        return [
            UsageRow(
                id=row["id"],
                seq=row["seq"],
                usage=usage_from_dict(json.loads(row["usage"])),
                adjustment=bool(row["adjustment"]),
                entry_id=row["entry_id"],
                details=None if row["details"] is None else json.loads(row["details"]),
            )
            for row in rows
        ]

    async def get_stats(self, context: Context) -> SessionStats:
        row = self._conn.execute(
            "SELECT message_count, usage FROM sessions WHERE id=?", (self._session_id,)
        ).fetchone()
        if row is None:  # pragma: no cover - 会话行在 __init__ 建好
            raise SessionCorruptError(f"missing sessions row: {self._session_id}")
        return SessionStats(
            message_count=int(row["message_count"]),
            usage=usage_from_dict(json.loads(row["usage"])),
        )

    async def close(self, context: Context) -> None:
        # 只有自建连接的实例才关库；attach 出来的会话视图不能把共享连接关掉。
        if self._owns_db:
            self._db.close()


def _like_prefix(prefix: str) -> str:
    # 转义 LIKE 通配符，做前缀匹配。
    escaped = prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return escaped + "%"


class _SqliteValidationView:
    """给 validate_committed_writes 的 SQL EXISTS 校验视图（按会话作用域）。"""

    def __init__(self, conn: sqlite3.Connection, session_id: str) -> None:
        self._conn = conn
        self._session_id = session_id

    def has_entry_or_usage_id(self, id: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM entries WHERE session_id=? AND id=? "
            "UNION ALL SELECT 1 FROM usage WHERE session_id=? AND id=? LIMIT 1",
            (self._session_id, id, self._session_id, id),
        ).fetchone()
        return row is not None

    def has_entry_id(self, id: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM entries WHERE session_id=? AND id=? LIMIT 1",
            (self._session_id, id),
        ).fetchone()
        return row is not None


__all__ = ["SqliteStorage", "SqliteDatabase", "STORAGE_VERSION"]
