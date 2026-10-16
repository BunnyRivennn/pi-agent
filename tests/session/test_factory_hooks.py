"""E1 端到端：before_compaction hook 的 decline / 自带摘要两条路径。"""

from __future__ import annotations

import asyncio
from typing import Any

from pi_agent.agent_core import (
    Agent,
    AgentLoopConfig,
    AssistantMessageEventStream,
    HookRegistry,
    LlmContext,
    Model,
    TextContent,
)
from pi_agent.agent_core.compaction import CompactionSettings
from pi_agent.agent_core.types import AssistantMessage, StreamFn, Usage
from pi_agent.session import (
    DEFAULT_BRANCH,
    Context,
    EntryScan,
    InMemorySessionRepo,
    create_agent_session,
)


def long_reply_stream_fn(text: str) -> StreamFn:
    """固定长回复的 mock stream_fn（与 test_factory_compaction 同形）。"""

    def stream_fn(
        _model: Model,
        _context: LlmContext,
        _config: AgentLoopConfig,
        _abort_event: asyncio.Event | None,
    ) -> AssistantMessageEventStream:
        stream = AssistantMessageEventStream()

        async def _emit() -> None:
            await asyncio.sleep(0)
            stream.push(
                {
                    "type": "done",
                    "reason": "stop",
                    "message": AssistantMessage(
                        content=[TextContent(text=text)],
                        api="mock-api",
                        provider="mock-provider",
                        model="mock-model",
                        stop_reason="stop",
                        usage=Usage(),
                    ),
                }
            )

        asyncio.create_task(_emit())
        return stream

    return stream_fn


_SMALL = CompactionSettings(enabled=True, reserve_tokens=0, keep_recent_tokens=5)
_TURNS = ["问题一", "问题二", "问题三", "问题四"]


async def _compaction_entries(session: Any, ctx: Context) -> list[Any]:
    entries = await session.find_entries(EntryScan(order="asc"), ctx)
    return [e for e in entries if e.type == "compaction"]


async def _build(ctx: Context, hooks: HookRegistry, summarize_calls: list[str]):
    async def fake_summarize(transcript: str) -> str:
        summarize_calls.append(transcript)
        return "【LLM摘要】"

    agent = Agent(stream_fn=long_reply_stream_fn("回答" * 200))
    return await create_agent_session(
        agent,
        InMemorySessionRepo(),
        ctx,
        summarize=fake_summarize,
        context_window=500,
        compaction_settings=_SMALL,
        hooks=hooks,
    )


# ---------------------------------------------------------------------------
# decline：本轮不压缩
# ---------------------------------------------------------------------------


async def test_hook_decline_则不压缩且不调用summarize() -> None:
    ctx = Context()
    hooks = HookRegistry()
    calls: list[str] = []
    hooks.on("before_compaction", lambda _e, _a: {"decline": True})

    sess = await _build(ctx, hooks, calls)
    for p in _TURNS:
        await sess.prompt(p)

    assert await _compaction_entries(sess.session, ctx) == []
    assert calls == []  # decline 在 LLM 调用之前，省掉了那次摘要


# ---------------------------------------------------------------------------
# 自带摘要：from_hook=True
# ---------------------------------------------------------------------------


async def test_hook_自带摘要_落库from_hook为真() -> None:
    ctx = Context()
    hooks = HookRegistry()
    calls: list[str] = []
    hooks.on(
        "before_compaction",
        lambda _e, _a: {"compaction": {"summary": "【扩展摘要】自定义压缩"}},
    )

    sess = await _build(ctx, hooks, calls)
    for p in _TURNS:
        await sess.prompt(p)

    compactions = await _compaction_entries(sess.session, ctx)
    assert compactions, "应至少压缩一次"
    assert compactions[0].summary == "【扩展摘要】自定义压缩"
    assert compactions[0].from_hook is True
    assert calls == []  # hook 给了摘要，跳过 LLM 调用


async def test_hook_看到真实的transcript和tokens() -> None:
    """hook 要替我们写摘要，就得看到被摘要的内容。"""
    ctx = Context()
    hooks = HookRegistry()
    seen: list[dict[str, Any]] = []

    def observer(event: dict[str, Any], _abort: asyncio.Event | None):
        seen.append(event)
        return {"compaction": {"summary": "摘要"}}

    hooks.on("before_compaction", observer)
    sess = await _build(ctx, hooks, [])
    for p in _TURNS:
        await sess.prompt(p)

    assert seen, "hook 应被调用"
    event = seen[0]
    # 对齐官方 before_compaction 契约：reason + preparation + lane/run_id
    assert event["reason"] == "threshold"
    assert event["lane"] == DEFAULT_BRANCH
    assert "run_id" in event
    prep = event["preparation"]
    assert prep["branch"] == DEFAULT_BRANCH
    assert prep["tokens_before"] > 0
    assert prep["context_window"] == 500
    assert prep["transcript"]  # 非空序列化文本
    assert prep["messages_to_summarize"]  # 被摘要掉的 head
    assert prep["retained_tail"] is not None


# ---------------------------------------------------------------------------
# 弃权与零影响
# ---------------------------------------------------------------------------


async def test_hook_弃权则回落到LLM摘要() -> None:
    ctx = Context()
    hooks = HookRegistry()
    calls: list[str] = []
    hooks.on("before_compaction", lambda _e, _a: None)

    sess = await _build(ctx, hooks, calls)
    for p in _TURNS:
        await sess.prompt(p)

    compactions = await _compaction_entries(sess.session, ctx)
    assert compactions
    assert compactions[0].summary == "【LLM摘要】"
    assert compactions[0].from_hook is False
    assert calls, "弃权后应回落到注入的 summarize"


async def test_hook_报错则回落到LLM摘要_不中断压缩() -> None:
    """before_compaction 是 fail-open：hook 坏了不该让压缩整体失败。"""
    ctx = Context()
    reported: list[str] = []
    hooks = HookRegistry(report_error=lambda exc, _n, _e: reported.append(str(exc)))
    calls: list[str] = []

    def broken(_event: dict[str, Any], _abort: asyncio.Event | None):
        raise RuntimeError("hook 内部错误")

    hooks.on("before_compaction", broken)
    sess = await _build(ctx, hooks, calls)
    for p in _TURNS:
        await sess.prompt(p)

    compactions = await _compaction_entries(sess.session, ctx)
    assert compactions
    assert compactions[0].summary == "【LLM摘要】"
    assert compactions[0].from_hook is False
    assert "hook 内部错误" in reported[0]


async def test_不挂hook时行为与之前一致() -> None:
    ctx = Context()
    calls: list[str] = []
    sess = await _build(ctx, HookRegistry(), calls)
    for p in _TURNS:
        await sess.prompt(p)

    compactions = await _compaction_entries(sess.session, ctx)
    assert compactions
    assert compactions[0].from_hook is False
    assert calls
