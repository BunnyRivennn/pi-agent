from __future__ import annotations

import os

import pytest

from pi_agent.agent_core.types import UserMessage
from pi_agent.session import (
    DEFAULT_BRANCH,
    BranchScan,
    Context,
    InMemorySessionRepo,
    MessageEntry,
    SqliteSessionRepo,
)


def user_msg(text: str) -> UserMessage:
    return UserMessage(content=text, timestamp=0)


# ---------------------------------------------------------------------------
# InMemorySessionRepo
# ---------------------------------------------------------------------------


async def test_inmemory_create_自动建默认分支() -> None:
    ctx = Context()
    repo = InMemorySessionRepo()
    s = await repo.create(ctx)
    # 默认分支已建（tip=None，非缺失）
    tip = await s.get_branch_tip(DEFAULT_BRANCH, ctx)
    print("tip=", tip)
    assert tip is None
    await repo.close(ctx)


async def test_inmemory_open同一实例可续读() -> None:
    ctx = Context()
    repo = InMemorySessionRepo()
    s1 = await repo.create(ctx)
    sid = s1.metadata.id
    branch = s1._get_or_create_branch(DEFAULT_BRANCH)
    eid = await branch.append_message(user_msg("hi"), ctx)

    s2 = await repo.open(sid, ctx)
    assert s2 is not None
    got = await s2.get_entry(eid, ctx)
    assert isinstance(got, MessageEntry)
    await repo.close(ctx)


async def test_inmemory_list_and_delete() -> None:
    ctx = Context()
    repo = InMemorySessionRepo()
    s = await repo.create(ctx)
    sid = s.metadata.id
    ids = await repo.list_ids(ctx)
    assert sid in ids
    assert await repo.delete(sid, ctx) is True
    assert await repo.delete(sid, ctx) is False
    assert sid not in await repo.list_ids(ctx)
    await repo.close(ctx)


# ---------------------------------------------------------------------------
# SqliteSessionRepo：落盘 → 重开续读
# ---------------------------------------------------------------------------


async def test_sqlite_落盘重开续读(tmp_path: object) -> None:
    ctx = Context()
    db = os.path.join(str(tmp_path), "sessions.db")

    repo1 = SqliteSessionRepo(db)
    s1 = await repo1.create(ctx)
    sid = s1.metadata.id
    branch = s1._get_or_create_branch(DEFAULT_BRANCH)
    id1 = await branch.append_message(user_msg("hello"), ctx)
    id2 = await branch.append_message(user_msg("world"), ctx)
    await s1.close(ctx)
    await repo1.close(ctx)

    # 全部会话共用这一个 db 文件
    assert os.path.exists(db)

    # 新 repo 实例重开同一个 db
    repo2 = SqliteSessionRepo(db)
    assert sid in await repo2.list_ids(ctx)
    s2 = await repo2.open(sid, ctx)
    assert s2 is not None
    tip = await s2.get_branch_tip(DEFAULT_BRANCH, ctx)
    assert tip == id2
    path = await s2.scan_branch(BranchScan(start=id2, order="oldestFirst"), ctx)
    assert [e.id for e in path] == [id1, id2]
    await s2.close(ctx)
    await repo2.close(ctx)


async def test_sqlite_多会话共一个db互不串数据(tmp_path: object) -> None:
    """单库多会话：两个会话写在同一文件里，读回来必须各管各的。"""
    ctx = Context()
    db = os.path.join(str(tmp_path), "sessions.db")
    repo = SqliteSessionRepo(db)

    sa = await repo.create(ctx)
    sb = await repo.create(ctx)
    assert sa.metadata.id != sb.metadata.id

    a_id = await sa._get_or_create_branch(DEFAULT_BRANCH).append_message(
        user_msg("会话 A 的消息"), ctx
    )
    b_id = await sb._get_or_create_branch(DEFAULT_BRANCH).append_message(
        user_msg("会话 B 的消息"), ctx
    )

    # 各自的 tip 不串
    assert await sa.get_branch_tip(DEFAULT_BRANCH, ctx) == a_id
    assert await sb.get_branch_tip(DEFAULT_BRANCH, ctx) == b_id

    # 各自的 entry 不串：A 看不到 B 的 entry
    assert await sa.get_entries([b_id], ctx) == {}
    assert await sb.get_entries([a_id], ctx) == {}

    # 各自的 stats 不串
    assert (await sa.get_stats(ctx)).message_count == 1
    assert (await sb.get_stats(ctx)).message_count == 1

    # 只有一个 db 文件，却有两个会话
    assert len([n for n in os.listdir(str(tmp_path)) if n.endswith(".db")]) == 1
    assert len(await repo.list_ids(ctx)) == 2
    await repo.close(ctx)


async def test_sqlite_list_ids按创建时间倒序(tmp_path: object) -> None:
    """新建的在前——这是单库换来的能力，扫目录时只能拿到字母序。"""
    ctx = Context()
    repo = SqliteSessionRepo(os.path.join(str(tmp_path), "sessions.db"))
    ids = []
    for _ in range(3):
        s = await repo.create(ctx)
        ids.append(s.metadata.id)
    listed = await repo.list_ids(ctx)
    assert listed == list(reversed(ids))
    await repo.close(ctx)


async def test_sqlite_open不存在返回None(tmp_path: object) -> None:
    ctx = Context()
    repo = SqliteSessionRepo(os.path.join(str(tmp_path), "s.db"))
    assert await repo.open("0192f000-0000-7000-8000-000000000000", ctx) is None
    await repo.close(ctx)


async def test_sqlite_delete清理会话数据(tmp_path: object) -> None:
    ctx = Context()
    repo = SqliteSessionRepo(os.path.join(str(tmp_path), "s.db"))
    keep = await repo.create(ctx)
    keep_id = keep.metadata.id
    keep_msg = await keep._get_or_create_branch(DEFAULT_BRANCH).append_message(
        user_msg("留着"), ctx
    )

    s = await repo.create(ctx)
    sid = s.metadata.id
    await s._get_or_create_branch(DEFAULT_BRANCH).append_message(user_msg("删掉"), ctx)
    await s.close(ctx)

    assert await repo.delete(sid, ctx) is True
    assert sid not in await repo.list_ids(ctx)
    assert await repo.open(sid, ctx) is None
    assert await repo.delete(sid, ctx) is False

    # 删一个会话不能误伤同库里的另一个
    assert keep_id in await repo.list_ids(ctx)
    assert set((await keep.get_entries([keep_msg], ctx)).keys()) == {keep_msg}
    await repo.close(ctx)


async def test_sqlite_重复create抛异常(tmp_path: object) -> None:
    ctx = Context()
    repo = SqliteSessionRepo(os.path.join(str(tmp_path), "s.db"))
    s = await repo.create(ctx, session_id="0192f000-0000-7000-8000-000000000000")
    await s.close(ctx)
    with pytest.raises(ValueError):
        await repo.create(ctx, session_id="0192f000-0000-7000-8000-000000000000")
    await repo.close(ctx)
