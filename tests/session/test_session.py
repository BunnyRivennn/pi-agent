from __future__ import annotations

import asyncio

import pytest

from pi_agent.agent_core.types import AssistantMessage, UserMessage
from pi_agent.session import (
    DEFAULT_BRANCH,
    BranchScan,
    Context,
    InMemoryStorage,
    MutationLine,
    SessionBranchExistsError,
    SessionClosedError,
    SessionInvariantError,
    SessionMetadata,
    SessionMutation,
    StorageBackedSession,
    branch_tip,
    build_context_entries,
    build_session_context,
    set_value,
)
from pi_agent.session.types import CompactionEntry, Entry, MessageEntry


def make_session() -> StorageBackedSession:
    storage = InMemoryStorage()
    meta = SessionMetadata(id="0192f000-0000-7000-8000-000000000000")
    return StorageBackedSession(meta, storage)


async def seed_main_branch(s: StorageBackedSession, ctx: Context) -> None:
    # 建默认分支（tip=None），append 才有落点。
    async def job(m: SessionMutation, c: Context) -> None:
        await m.commit([set_value(branch_tip(DEFAULT_BRANCH), None)], c)

    await s.mutate(job, ctx)


def user_msg(text: str) -> UserMessage:
    return UserMessage(content=text, timestamp=0)


# ---------------------------------------------------------------------------
# MutationLine：串行化
# ---------------------------------------------------------------------------


async def test_mutation_line_串行执行不交错() -> None:
    line = MutationLine()
    order: list[str] = []

    async def op(tag: str, delay: float) -> str:
        order.append(f"{tag}-start")
        await asyncio.sleep(delay)
        order.append(f"{tag}-end")
        return tag

    # 先发慢的，再发快的；串行应保证 a 完整结束后 b 才开始。
    results = await asyncio.gather(
        line.run(lambda: op("a", 0.02)),
        line.run(lambda: op("b", 0.0)),
    )
    print("order=", order, "results=", results)
    assert order == ["a-start", "a-end", "b-start", "b-end"]
    assert list(results) == ["a", "b"]


async def test_mutation_line_seal后拒绝新作业() -> None:
    line = MutationLine()
    await line.seal(RuntimeError("closed"))
    with pytest.raises(RuntimeError):
        await line.run(lambda: _return(1))


async def _return(x: int) -> int:
    return x


# ---------------------------------------------------------------------------
# append_to_branch：原子提交 entry + tip
# ---------------------------------------------------------------------------


async def test_append_message_更新tip并链接父() -> None:
    ctx = Context()
    s = make_session()
    await seed_main_branch(s, ctx)
    branch = s._get_or_create_branch(DEFAULT_BRANCH)

    id1 = await branch.append_message(user_msg("hello"), ctx)
    id2 = await branch.append_message(user_msg("world"), ctx)
    print("ids=", id1, id2)

    tip = await s.get_branch_tip(DEFAULT_BRANCH, ctx)
    assert tip == id2

    e2 = await s.get_entry(id2, ctx)
    assert isinstance(e2, MessageEntry)
    assert e2.parent_id == id1

    # scan_branch 从 tip 上溯应得 [id1, id2]
    path = await s.scan_branch(BranchScan(start=id2, order="oldestFirst"), ctx)
    assert [e.id for e in path] == [id1, id2]


async def test_未建分支时append抛不变量错误() -> None:
    ctx = Context()
    s = make_session()  # 未 seed main
    branch = s._get_or_create_branch(DEFAULT_BRANCH)
    with pytest.raises(SessionInvariantError):
        await branch.append_message(user_msg("x"), ctx)


# ---------------------------------------------------------------------------
# create_branch
# ---------------------------------------------------------------------------


async def test_create_branch_已存在抛异常() -> None:
    ctx = Context()
    s = make_session()
    await s.create_branch("feature", None, ctx)
    with pytest.raises(SessionBranchExistsError):
        await s.create_branch("feature", None, ctx)


async def test_branch_不存在返回None() -> None:
    ctx = Context()
    s = make_session()
    assert await s.branch("nope", ctx) is None


# ---------------------------------------------------------------------------
# close
# ---------------------------------------------------------------------------


async def test_close后写操作被拒() -> None:
    ctx = Context()
    s = make_session()
    await seed_main_branch(s, ctx)
    await s.close(ctx)
    with pytest.raises(SessionClosedError):
        await s.set_value(branch_tip("x"), "v", ctx)


# ---------------------------------------------------------------------------
# build_context：压缩截断
# ---------------------------------------------------------------------------


def test_build_context_entries_无压缩返回全部() -> None:
    entries = [
        MessageEntry(id="a", parent_id=None, seq=1, message=user_msg("1")),
        MessageEntry(id="b", parent_id="a", seq=2, message=user_msg("2")),
    ]
    out = build_context_entries(entries)
    assert [e.id for e in out] == ["a", "b"]


def test_build_context_entries_从最后压缩点截断() -> None:
    entries: list[Entry] = [
        MessageEntry(id="a", parent_id=None, seq=1, message=user_msg("old")),
        CompactionEntry(
            id="c", parent_id="a", seq=2, summary="sum", tokens_before=100,
            retained_tail=[],
        ),
        MessageEntry(id="b", parent_id="c", seq=3, message=user_msg("new")),
    ]
    out = build_context_entries(entries)
    print("out=", [e.id for e in out])
    assert [e.id for e in out] == ["c", "b"]


async def test_build_session_context_压缩后含摘要与新消息() -> None:
    entries: list[Entry] = [
        MessageEntry(id="a", parent_id=None, seq=1, message=user_msg("old")),
        CompactionEntry(
            id="c", parent_id="a", seq=2, summary="SUMMARY", tokens_before=100,
            retained_tail=[],
        ),
        MessageEntry(id="b", parent_id="c", seq=3, message=user_msg("new")),
    ]
    msgs = await build_session_context(entries, None)
    print("msgs=", [type(m).__name__ for m in msgs])
    # 第一条是压缩摘要（UserMessage 前缀），最后是 new
    assert any("SUMMARY" in m.content for m in msgs if isinstance(m, UserMessage))
    assert any(
        isinstance(m, UserMessage) and m.content == "new" for m in msgs
    )


async def test_build_session_context_跳过error的assistant() -> None:
    err = AssistantMessage(
        content=[], api="a", provider="p", model="m", stop_reason="error"
    )
    entries: list[Entry] = [
        MessageEntry(id="a", parent_id=None, seq=1, message=user_msg("ok")),
        MessageEntry(id="b", parent_id="a", seq=2, message=err),
    ]
    msgs = await build_session_context(entries, None)
    print("count=", len(msgs))
    assert len(msgs) == 1
    assert isinstance(msgs[0], UserMessage)
