"""存储层：Storage 契约 + 提交管线 + 后端实现。

分层（依赖自上而下，无循环）：
- ``storage_types``：``Storage`` Protocol、``Write`` 联合、扫描查询、``CommitResult``。
- ``commit``：seq 分配 + 写入校验（纯函数，后端无关，两个后端共用）。
- ``memory``：``InMemoryStorage``（测试/临时会话）。
- ``sqlite``：``SqliteStorage``（落盘，WAL + 事务化 commit + 持久 seq 计数器）。

新增后端只需实现 ``storage_types.Storage``，并复用 ``commit`` 的
``prepare_storage_commit`` / ``validate_committed_writes``。
"""

from __future__ import annotations

from .commit import (
    CommitError,
    CommittedEntryWrite,
    CommittedListWrite,
    CommittedUsageWrite,
    CommittedValueWrite,
    CommittedWrite,
    CommitValidationState,
    PreparedCommit,
    add_usage,
    commit_write,
    committed_seq,
    insert_entry,
    insert_usage,
    prepare_storage_commit,
    validate_committed_writes,
)
from .memory import InMemoryStorage, InMemoryStorageState
from .sqlite import STORAGE_VERSION, SqliteStorage
from .storage_types import (
    BranchScan,
    CommitResult,
    Context,
    EntryCursor,
    EntryScan,
    EntryStructure,
    EntryWrite,
    SessionStats,
    Storage,
    UsageRow,
    UsageScan,
    UsageWrite,
    Write,
)

__all__ = [
    # 契约与类型
    "Storage",
    "Context",
    "Write",
    "EntryWrite",
    "UsageWrite",
    "UsageRow",
    "CommitResult",
    "SessionStats",
    "EntryStructure",
    "EntryCursor",
    "BranchScan",
    "EntryScan",
    "UsageScan",
    # 提交管线
    "CommitError",
    "CommitValidationState",
    "PreparedCommit",
    "CommittedWrite",
    "CommittedEntryWrite",
    "CommittedUsageWrite",
    "CommittedValueWrite",
    "CommittedListWrite",
    "prepare_storage_commit",
    "validate_committed_writes",
    "committed_seq",
    "insert_entry",
    "insert_usage",
    "commit_write",
    "add_usage",
    # 后端
    "InMemoryStorage",
    "InMemoryStorageState",
    "SqliteStorage",
    "STORAGE_VERSION",
]
