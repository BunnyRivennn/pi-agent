from __future__ import annotations

import pytest

from pi_agent.agent_core.types import Usage, UsageCost, UserMessage
from pi_agent.session import (
    BranchScan,
    CommitError,
    Context,
    EntryScan,
    InMemoryStorage,
    JsonValue,
    MessageEntry,
    SessionCorruptError,
    UsageScan,
    Value,
    ValueList,
    append_list,
    insert_entry,
    insert_usage,
    lane_config,
    list_,
    set_value,
    value,
)


def make_storage() -> InMemoryStorage:
    return InMemoryStorage()


def msg_entry(entry_id: str, parent: str | None, text: str) -> MessageEntry:
    # NewEntry 语义：不传 seq/timestamp，由存储层分配。
    return MessageEntry(
        id=entry_id,
        parent_id=parent,
        message=UserMessage(content=text, timestamp=0),
    )


def usage(n: int) -> Usage:
    return Usage(input=n, output=n, total_tokens=2 * n, cost=UsageCost(total=0.1 * n))


# ---------------------------------------------------------------------------
# commit：seq 分配 + stats
# ---------------------------------------------------------------------------


async def test_commit_分配连续seq_从1起() -> None:
    s = make_storage()
    ctx = Context()
    r = await s.commit(
        [insert_entry(msg_entry("a", None, "hi")), insert_entry(msg_entry("b", "a", "yo"))],
        ctx,
    )
    print("first_seq=", r.first_seq, "seqs=", r.seqs)
    assert r.first_seq == 1
    assert r.seqs == [1, 2]
    assert r.stats.message_count == 2


async def test_commit_跨批次seq递增() -> None:
    s = make_storage()
    ctx = Context()
    await s.commit([insert_entry(msg_entry("a", None, "1"))], ctx)
    r2 = await s.commit([insert_entry(msg_entry("b", "a", "2"))], ctx)
    print("r2.first_seq=", r2.first_seq)
    assert r2.first_seq == 2
    assert r2.seqs == [2]


async def test_commit_usage累加进stats() -> None:
    s = make_storage()
    ctx = Context()
    r = await s.commit(
        [insert_usage("u1", usage(10)), insert_usage("u2", usage(5))], ctx
    )
    print("stats.usage=", r.stats.usage)
    assert r.stats.usage.input == 15
    assert r.stats.usage.total_tokens == 30
    assert r.stats.message_count == 0


# ---------------------------------------------------------------------------
# entry 读取 + roundtrip 一致
# ---------------------------------------------------------------------------


async def test_get_entries_返回已分配seq() -> None:
    s = make_storage()
    ctx = Context()
    await s.commit([insert_entry(msg_entry("a", None, "hi"))], ctx)
    got = await s.get_entries(["a", "missing"], ctx)
    print("got=", got)
    assert set(got) == {"a"}
    assert got["a"].seq == 1
    assert got["a"].timestamp > 0


# ---------------------------------------------------------------------------
# scan_branch：沿 parent_id 上溯
# ---------------------------------------------------------------------------


async def test_scan_branch_oldest_first_是根到叶路径() -> None:
    s = make_storage()
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
    ids = [e.id for e in path]
    print("path=", ids)
    assert ids == ["a", "b", "c"]


async def test_scan_branch_未知start抛异常() -> None:
    s = make_storage()
    ctx = Context()
    with pytest.raises(SessionCorruptError):
        await s.scan_branch(BranchScan(start="nope"), ctx)


# ---------------------------------------------------------------------------
# scan_entries：按 seq 序
# ---------------------------------------------------------------------------


async def test_scan_entries_desc序() -> None:
    s = make_storage()
    ctx = Context()
    await s.commit(
        [
            insert_entry(msg_entry("a", None, "1")),
            insert_entry(msg_entry("b", "a", "2")),
        ],
        ctx,
    )
    desc = await s.scan_entries(EntryScan(order="desc"), ctx)
    print("desc=", [e.id for e in desc])
    assert [e.id for e in desc] == ["b", "a"]


# ---------------------------------------------------------------------------
# values + lists
# ---------------------------------------------------------------------------


async def test_value_set_get() -> None:
    s = make_storage()
    ctx = Context()
    cfg: JsonValue = {"model": "m", "thinking": "off"}
    await s.commit([set_value(lane_config("main"), cfg)], ctx)
    stored = await s.get_value(lane_config("main"), ctx)
    print("stored=", stored)
    assert stored is not None
    assert stored.value == {"model": "m", "thinking": "off"}


async def test_list_append_read_保序() -> None:
    s = make_storage()
    ctx = Context()
    addr: ValueList[str] = list_("pi.log")
    await s.commit([append_list(addr, "a"), append_list(addr, "b")], ctx)
    await s.commit([append_list(addr, "c")], ctx)
    els = await s.read_list(addr, None, ctx)
    print("els=", [e.value for e in els])
    assert [e.value for e in els] == ["a", "b", "c"]


async def test_scan_values_前缀过滤() -> None:
    s = make_storage()
    ctx = Context()
    await s.commit(
        [set_value(value("pi.x", "k1"), 1), set_value(value("pi.x", "k2"), 2)], ctx
    )
    prefix: Value[int] = value("pi.x")
    rows = await s.scan_values(prefix, ctx)
    print("rows=", [(r.address.key, r.value) for r in rows])
    assert [r.address.key for r in rows] == ["k1", "k2"]


async def test_scan_usage() -> None:
    s = make_storage()
    ctx = Context()
    await s.commit([insert_usage("u1", usage(3))], ctx)
    rows = await s.scan_usage(UsageScan(), ctx)
    print("rows=", rows)
    assert len(rows) == 1
    assert rows[0].usage.input == 3


# ---------------------------------------------------------------------------
# 校验失败：整批不落库
# ---------------------------------------------------------------------------


async def test_commit_重复id抛异常() -> None:
    s = make_storage()
    ctx = Context()
    await s.commit([insert_entry(msg_entry("a", None, "1"))], ctx)
    with pytest.raises(CommitError):
        await s.commit([insert_entry(msg_entry("a", "a", "dup"))], ctx)


async def test_commit_缺失父条目抛异常() -> None:
    s = make_storage()
    ctx = Context()
    with pytest.raises(CommitError):
        await s.commit([insert_entry(msg_entry("b", "ghost", "x"))], ctx)


async def test_commit_同事务内父先出现允许() -> None:
    s = make_storage()
    ctx = Context()
    # a、b 在同一批，b 的父是 a：应允许。
    r = await s.commit(
        [insert_entry(msg_entry("a", None, "1")), insert_entry(msg_entry("b", "a", "2"))],
        ctx,
    )
    assert r.seqs == [1, 2]


async def test_get_stats() -> None:
    s = make_storage()
    ctx = Context()
    await s.commit([insert_entry(msg_entry("a", None, "1"))], ctx)
    stats = await s.get_stats(ctx)
    print("stats=", stats)
    assert stats.message_count == 1
