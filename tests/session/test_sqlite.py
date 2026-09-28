from __future__ import annotations

import os

import pytest

from pi_agent.agent_core.types import Usage, UsageCost, UserMessage
from pi_agent.session import (
    BranchScan,
    CommitError,
    Context,
    EntryScan,
    JsonValue,
    MessageEntry,
    SessionCorruptError,
    SqliteStorage,
    UsageScan,
    Value,
    ValueList,
    append_list,
    delete_value,
    insert_entry,
    insert_usage,
    lane_config,
    list_,
    set_value,
    value,
)


def msg_entry(entry_id: str, parent: str | None, text: str) -> MessageEntry:
    """构造一个简单的 MessageEntry 用于测试。"""
    return MessageEntry(
        id=entry_id,
        parent_id=parent,
        message=UserMessage(content=text, timestamp=0),
    )


def usage(n: int) -> Usage:
    """构造一个指定输入输出量的 Usage 对象用于测试。"""
    return Usage(input=n, output=n, total_tokens=2 * n, cost=UsageCost(total=0.1 * n))


# ---------------------------------------------------------------------------
# 行为契约（与 InMemoryStorage 一致）
# ---------------------------------------------------------------------------


async def test_commit_assigns_seq_starting_from_one() -> None:
    """测试 commit 分配的 seq 从 1 开始递增。"""
    s = SqliteStorage()
    ctx = Context()
    r = await s.commit(
        [insert_entry(msg_entry("a", None, "hi")), insert_entry(msg_entry("b", "a", "y"))],
        ctx,
    )
    print("seqs=", r.seqs)
    assert r.first_seq == 1
    assert r.seqs == [1, 2]
    assert r.stats.message_count == 2
    await s.close(ctx)


async def test_scan_branch_oldest_first() -> None:
    """测试扫描分支时按从旧到新的顺序返回。"""
    s = SqliteStorage()
    ctx = Context()
    await s.commit(
        [
            insert_entry(msg_entry("a", None, "1")),
            insert_entry(msg_entry("b", "a", "2")),
            insert_entry(msg_entry("c", "b", "3")),
        ],
        ctx,
    )
    path = await s.scan_branch(BranchScan(start="c", order="oldestFirst"), ctx)
    print("path=", [e.id for e in path])
    assert [e.id for e in path] == ["a", "b", "c"]
    await s.close(ctx)


async def test_scan_entries_desc() -> None:
    """测试扫描条目时按降序（从新到旧）返回。"""
    s = SqliteStorage()
    ctx = Context()
    await s.commit(
        [insert_entry(msg_entry("a", None, "1")), insert_entry(msg_entry("b", "a", "2"))],
        ctx,
    )
    desc = await s.scan_entries(EntryScan(order="desc"), ctx)
    assert [e.id for e in desc] == ["b", "a"]
    await s.close(ctx)


async def test_value_and_list() -> None:
    """测试值的设置与获取以及列表的追加与读取。"""
    s = SqliteStorage()
    ctx = Context()
    cfg: JsonValue = {"model": "m"}
    await s.commit([set_value(lane_config("main"), cfg)], ctx)
    stored = await s.get_value(lane_config("main"), ctx)
    assert stored is not None and stored.value == {"model": "m"}

    addr: ValueList[str] = list_("pi.log")
    await s.commit([append_list(addr, "a"), append_list(addr, "b")], ctx)
    await s.commit([append_list(addr, "c")], ctx)
    els = await s.read_list(addr, None, ctx)
    assert [e.value for e in els] == ["a", "b", "c"]
    await s.close(ctx)


async def test_scan_values_prefix_filter() -> None:
    """测试按命名空间前缀扫描值。"""
    s = SqliteStorage()
    ctx = Context()
    prefix: Value[int] = value("pi.x")
    await s.commit(
        [set_value(value("pi.x", "k1"), 1), set_value(value("pi.x", "k2"), 2)], ctx
    )
    rows = await s.scan_values(prefix, ctx)
    assert [r.address.key for r in rows] == ["k1", "k2"]
    await s.close(ctx)


async def test_value_delete() -> None:
    """测试删除值后无法再获取。"""
    s = SqliteStorage()
    ctx = Context()
    await s.commit([set_value(value("pi.x", "k"), 1)], ctx)
    await s.commit([delete_value(value("pi.x", "k"))], ctx)
    assert await s.get_value(value("pi.x", "k"), ctx) is None
    await s.close(ctx)


async def test_scan_usage() -> None:
    """测试用量记录的扫描与统计汇总。"""
    s = SqliteStorage()
    ctx = Context()
    await s.commit([insert_usage("u1", usage(3)), insert_usage("u2", usage(4))], ctx)
    rows = await s.scan_usage(UsageScan(), ctx)
    assert [r.usage.input for r in rows] == [3, 4]
    stats = await s.get_stats(ctx)
    assert stats.usage.input == 7
    await s.close(ctx)


# ---------------------------------------------------------------------------
# 校验失败：整批不落库
# ---------------------------------------------------------------------------


async def test_duplicate_id_raises_error() -> None:
    """测试重复的条目 ID 会抛出 CommitError，整批不落库。"""
    s = SqliteStorage()
    ctx = Context()
    await s.commit([insert_entry(msg_entry("a", None, "1"))], ctx)
    with pytest.raises(CommitError):
        await s.commit([insert_entry(msg_entry("a", "a", "dup"))], ctx)
    await s.close(ctx)


async def test_missing_parent_raises_error() -> None:
    """测试引用不存在的父条目会抛出 CommitError，整批不落库。"""
    s = SqliteStorage()
    ctx = Context()
    with pytest.raises(CommitError):
        await s.commit([insert_entry(msg_entry("b", "ghost", "x"))], ctx)
    await s.close(ctx)


async def test_unknown_branch_start() -> None:
    """测试扫描不存在的分支起点会抛出 SessionCorruptError。"""
    s = SqliteStorage()
    ctx = Context()
    with pytest.raises(SessionCorruptError):
        await s.scan_branch(BranchScan(start="nope"), ctx)
    await s.close(ctx)


# ---------------------------------------------------------------------------
# 持久化专属：落盘 → 重开回放一致
# ---------------------------------------------------------------------------


async def test_persistence_reopen_consistency(tmp_path: object) -> None:
    """测试数据落盘后重新打开，回放结果一致。"""
    db = os.path.join(str(tmp_path), "sess.db")
    ctx = Context()

    s1 = SqliteStorage(db)
    await s1.commit(
        [
            insert_entry(msg_entry("a", None, "hello")),
            insert_entry(msg_entry("b", "a", "world")),
        ],
        ctx,
    )
    cfg: JsonValue = {"model": "m", "thinking": "off"}
    await s1.commit([set_value(lane_config("main"), cfg)], ctx)
    await s1.close(ctx)

    # 重开同一文件
    s2 = SqliteStorage(db)
    path = await s2.scan_branch(BranchScan(start="b", order="oldestFirst"), ctx)
    print("reopened path=", [e.id for e in path])
    assert [e.id for e in path] == ["a", "b"]
    # 消息内容 roundtrip 一致
    got = await s2.get_entries(["a"], ctx)
    entry = got["a"]
    assert isinstance(entry, MessageEntry)
    assert isinstance(entry.message, UserMessage)
    assert entry.message.content == "hello"
    # 配置恢复
    stored = await s2.get_value(lane_config("main"), ctx)
    assert stored is not None and stored.value == cfg
    # 统计恢复
    stats = await s2.get_stats(ctx)
    assert stats.message_count == 2
    await s2.close(ctx)


async def test_reopen_seq_continues_incrementing(tmp_path: object) -> None:
    """测试重新打开数据库后 seq 继续递增，不会重置。"""
    db = os.path.join(str(tmp_path), "seq.db")
    ctx = Context()
    s1 = SqliteStorage(db)
    await s1.commit([insert_entry(msg_entry("a", None, "1"))], ctx)
    await s1.close(ctx)

    s2 = SqliteStorage(db)
    r = await s2.commit([insert_entry(msg_entry("b", "a", "2"))], ctx)
    print("reopened first_seq=", r.first_seq)
    assert r.first_seq == 2
    await s2.close(ctx)


async def test_seq_never_regresses_after_delete() -> None:
    """删除持有最高 seq 的行之后，seq 必须继续前进、绝不重用。

    这正是旧的 ``MAX(seq)+1`` 方案的真实缺陷：删掉最高 seq 的行后 MAX 回退，
    下一次分配会重用已用过的 seq（实测旧方案给出 [1,2,3,2]）。
    持久计数器（session_stats.next_seq）只增不减，故为 [1,2,3,4]。
    """
    s = SqliteStorage()
    ctx = Context()
    r1 = await s.commit([set_value(value("pi.x", "a"), 1)], ctx)
    r2 = await s.commit([set_value(value("pi.x", "b"), 2)], ctx)
    # 删除 b —— 它持有当前最高 seq
    r3 = await s.commit([delete_value(value("pi.x", "b"))], ctx)
    # 再写一个新 key：绝不能拿回已用过的 seq
    r4 = await s.commit([set_value(value("pi.x", "c"), 3)], ctx)

    seqs = [r1.first_seq, r2.first_seq, r3.first_seq, r4.first_seq]
    print("allocated seqs=", seqs)
    assert seqs == [1, 2, 3, 4]
    assert all(b > a for a, b in zip(seqs, seqs[1:], strict=False))
    await s.close(ctx)


async def test_seq_counter_survives_reopen_without_rows(tmp_path: object) -> None:
    """计数器持久化：即使删空了所有 value 行，重开后 seq 也从计数器续起，
    不会因为表里没行而重置回 1。"""
    db = os.path.join(str(tmp_path), "counter.db")
    ctx = Context()
    s1 = SqliteStorage(db)
    await s1.commit([set_value(value("pi.x", "k"), 1)], ctx)
    await s1.commit([delete_value(value("pi.x", "k"))], ctx)  # 表被删空
    await s1.close(ctx)

    s2 = SqliteStorage(db)
    r = await s2.commit([set_value(value("pi.x", "fresh"), 9)], ctx)
    print("first_seq after reopen with empty tables=", r.first_seq)
    assert r.first_seq == 3  # 1=set, 2=delete, 3=本次
    await s2.close(ctx)
