"""D2 端到端压缩测试：多轮对话触发压缩 → CompactionEntry 落库、原始 entry 仍在。"""

from __future__ import annotations

import asyncio

from pi_agent.agent_core import (
    Agent,
    AgentLoopConfig,
    AssistantMessageEventStream,
    LlmContext,
    Model,
    TextContent,
)
from pi_agent.agent_core.compaction import CompactionSettings
from pi_agent.agent_core.types import AssistantMessage, StreamFn, Usage
from pi_agent.session import (
    Context,
    EntryScan,
    InMemorySessionRepo,
    create_agent_session,
)

# keep_recent 调到极小，使切点能落在对话后段、给摘要留出可压缩的 head。
_SMALL = CompactionSettings(enabled=True, reserve_tokens=0, keep_recent_tokens=5)


def _long_assistant(text: str) -> AssistantMessage:
    return AssistantMessage(
        content=[TextContent(text=text)],
        api="mock-api",
        provider="mock-provider",
        model="mock-model",
        stop_reason="stop",
        usage=Usage(),
    )


def long_reply_stream_fn(text: str) -> StreamFn:
    def stream_fn(
        _model: Model,
        _context: LlmContext,
        _config: AgentLoopConfig,
        _abort_event: asyncio.Event | None,
    ) -> AssistantMessageEventStream:
        stream = AssistantMessageEventStream()

        async def _emit() -> None:
            await asyncio.sleep(0)
            stream.push({"type": "done", "reason": "stop", "message": _long_assistant(text)})

        asyncio.create_task(_emit())
        return stream

    return stream_fn


async def _run_turns(sess: object, agent: Agent, prompts: list[str]) -> None:
    for p in prompts:
        # 走 AgentSession.prompt：它内部 await 落库 + 串行压缩检查。
        await sess.prompt(p)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# 触发压缩
# ---------------------------------------------------------------------------


async def test_多轮触发压缩_写入compaction_entry() -> None:
    ctx = Context()
    repo = InMemorySessionRepo()
    summarize_calls: list[str] = []

    async def fake_summarize(transcript: str) -> str:
        summarize_calls.append(transcript)
        return "【摘要】早前对话已压缩"

    # 长回复 + 极小 context_window → 几轮即超阈值。
    agent = Agent(stream_fn=long_reply_stream_fn("回答" * 200))
    sess = await create_agent_session(
        agent, repo, ctx,
        summarize=fake_summarize,
        context_window=500,
        compaction_settings=_SMALL,
    )
    sid = sess.session.metadata.id

    await _run_turns(sess, agent, ["问题一", "问题二", "问题三", "问题四"])

    # 应至少触发一次摘要
    print("summarize_calls=", len(summarize_calls))
    assert len(summarize_calls) >= 1

    # 库里应有 compaction entry
    entries = await sess.session.find_entries(EntryScan(order="asc"), ctx)
    kinds = [e.type for e in entries]
    print("entry kinds=", kinds)
    assert "compaction" in kinds

    await sess.dispose()

    # 重开：build_context 应从摘要处截断，恢复的首条是摘要标记的 UserMessage
    agent2 = Agent(stream_fn=long_reply_stream_fn("x"))
    sess2 = await create_agent_session(agent2, repo, ctx, session_id=sid)
    first = agent2.state.messages[0]
    print("restored first=", type(first).__name__, getattr(first, "content", None))
    assert "[Conversation Summary]" in str(getattr(first, "content", ""))
    await sess2.dispose()


async def test_原始entry压缩后仍在库中() -> None:
    ctx = Context()
    repo = InMemorySessionRepo()

    async def fake_summarize(_transcript: str) -> str:
        return "摘要"

    agent = Agent(stream_fn=long_reply_stream_fn("答" * 200))
    sess = await create_agent_session(
        agent, repo, ctx, summarize=fake_summarize, context_window=500,
        compaction_settings=_SMALL,
    )
    await _run_turns(sess, agent, ["a", "b", "c", "d"])

    entries = await sess.session.find_entries(EntryScan(order="asc"), ctx)
    message_entries = [e for e in entries if e.type == "message"]
    compaction_entries = [e for e in entries if e.type == "compaction"]
    print("messages=", len(message_entries), "compactions=", len(compaction_entries))
    # 压缩不删除原始 message entry —— 它们与 compaction entry 并存
    assert len(compaction_entries) >= 1
    assert len(message_entries) >= 4  # 至少最初几轮的原始消息还在

    await sess.dispose()


# ---------------------------------------------------------------------------
# 零变化：不注入 summarize 则不压缩
# ---------------------------------------------------------------------------


async def test_未注入summarize不压缩() -> None:
    ctx = Context()
    repo = InMemorySessionRepo()
    agent = Agent(stream_fn=long_reply_stream_fn("答" * 200))
    # 不传 summarize/context_window → compaction_enabled 为 False
    sess = await create_agent_session(agent, repo, ctx)
    assert sess.compaction_enabled is False
    await _run_turns(sess, agent, ["a", "b", "c", "d"])

    entries = await sess.session.find_entries(EntryScan(order="asc"), ctx)
    kinds = [e.type for e in entries]
    print("kinds=", kinds)
    assert "compaction" not in kinds
    await sess.dispose()


async def test_context_window缺失则不压缩() -> None:
    ctx = Context()
    repo = InMemorySessionRepo()

    async def fake_summarize(_t: str) -> str:
        return "s"

    agent = Agent(stream_fn=long_reply_stream_fn("答" * 200))
    # 只传 summarize、不传 context_window → 仍不启用
    sess = await create_agent_session(agent, repo, ctx, summarize=fake_summarize)
    assert sess.compaction_enabled is False
    await sess.dispose()


# ---------------------------------------------------------------------------
# 回归：压缩触发的幂等与防死循环
# ---------------------------------------------------------------------------


async def test_一轮只压缩一次() -> None:
    """回归：曾因在每条 message_end 上 fire-and-forget 触发压缩，
    导致一轮（user + assistant 两条消息）跑两次摘要、写出成对的 compaction entry。
    现在压缩在 prompt() 主流串行 await，且只看 assistant 消息。"""
    ctx = Context()
    repo = InMemorySessionRepo()
    calls: list[str] = []

    async def fake_summarize(transcript: str) -> str:
        calls.append(transcript)
        return "【摘要】"

    agent = Agent(stream_fn=long_reply_stream_fn("回答" * 200))
    sess = await create_agent_session(
        agent, repo, ctx,
        summarize=fake_summarize,
        context_window=500,
        compaction_settings=_SMALL,
    )

    await _run_turns(sess, agent, ["问题一", "问题二", "问题三", "问题四"])

    entries = await sess.session.find_entries(EntryScan(order="asc"), ctx)
    compactions = [e for e in entries if e.type == "compaction"]
    print("summarize_calls=", len(calls), "compaction_entries=", len(compactions))

    # 每次摘要调用恰好对应一个 compaction entry（无重复写入）
    assert len(compactions) == len(calls)
    # 四轮对话不可能产生超过四个 compaction（旧 bug 下会翻倍）
    assert len(compactions) <= 4

    # 不存在相邻的两个 compaction entry（旧 bug 的特征形状）
    kinds = [e.type for e in entries]
    assert "compaction" not in [
        kinds[i + 1] for i, k in enumerate(kinds[:-1]) if k == "compaction"
    ], f"出现相邻的重复 compaction: {kinds}"

    await sess.dispose()


async def test_压缩后下一轮不立即重压() -> None:
    """回归（对齐官方 assistantIsFromBeforeCompaction）：压缩保留的旧消息
    携带压缩前的陈旧 usage，不得被用来在刚压完后立刻再触发一次压缩。"""
    ctx = Context()
    repo = InMemorySessionRepo()
    calls: list[str] = []

    async def fake_summarize(transcript: str) -> str:
        calls.append(transcript)
        return "【摘要】"

    agent = Agent(stream_fn=long_reply_stream_fn("回答" * 200))
    sess = await create_agent_session(
        agent, repo, ctx,
        summarize=fake_summarize,
        context_window=500,
        compaction_settings=_SMALL,
    )

    # 跑到至少发生一次压缩
    await _run_turns(sess, agent, ["a", "b", "c", "d"])
    assert len(calls) >= 1

    # 直接对同一个已压缩过的状态再调一次检查：_last_assistant 已被取走清空，
    # 应直接返回 False，不产生新的摘要调用。
    before = len(calls)
    again = await sess._check_compaction()  # type: ignore[attr-defined]
    print("repeat check returned", again, "calls", before, "->", len(calls))
    assert again is False
    assert len(calls) == before

    await sess.dispose()
