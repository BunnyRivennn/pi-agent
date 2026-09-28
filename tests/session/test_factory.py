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
from pi_agent.agent_core.types import (
    AssistantMessage,
    StopReason,
    StreamFn,
    Usage,
    UserMessage,
)
from pi_agent.session import (
    DEFAULT_BRANCH,
    Context,
    InMemorySessionRepo,
    create_agent_session,
)


def make_assistant(text: str, *, stop_reason: StopReason = "stop") -> AssistantMessage:
    return AssistantMessage(
        content=[TextContent(text=text)],
        api="mock-api",
        provider="mock-provider",
        model="mock-model",
        stop_reason=stop_reason,
        usage=Usage(),
    )


def reply_stream_fn(text: str) -> StreamFn:
    """构造一个"回一句 text 就结束一轮"的假模型 stream_fn。"""

    def stream_fn(
        _model: Model,
        _context: LlmContext,
        _config: AgentLoopConfig,
        _abort_event: asyncio.Event | None,
    ) -> AssistantMessageEventStream:
        stream = AssistantMessageEventStream()

        async def _emit() -> None:
            await asyncio.sleep(0)
            stream.push({"type": "done", "reason": "stop", "message": make_assistant(text)})

        asyncio.create_task(_emit())
        return stream

    return stream_fn


# ---------------------------------------------------------------------------
# 核心验收：跑一轮落库 → 重开续聊消息一致
# ---------------------------------------------------------------------------


async def test_prompt落库_重开会话历史一致() -> None:
    ctx = Context()
    repo = InMemorySessionRepo()

    agent = Agent(stream_fn=reply_stream_fn("答复一"))
    sess = await create_agent_session(agent, repo, ctx)
    sid = sess.session.metadata.id

    await agent.prompt("问题一")
    await agent.wait_for_idle()
    await sess.flush()  # 等在途落库任务完成

    # agent 状态里应有 user + assistant 两条
    print("agent messages=", [type(m).__name__ for m in agent.state.messages])
    assert len(agent.state.messages) == 2
    await sess.dispose()

    # 重开同一会话：历史应被灌回新 agent
    agent2 = Agent(stream_fn=reply_stream_fn("答复二"))
    sess2 = await create_agent_session(agent2, repo, ctx, session_id=sid)
    print("restored=", [type(m).__name__ for m in agent2.state.messages])
    assert len(agent2.state.messages) == 2
    assert isinstance(agent2.state.messages[0], UserMessage)
    assert isinstance(agent2.state.messages[1], AssistantMessage)
    await sess2.dispose()


async def test_两轮prompt落库四条消息() -> None:
    ctx = Context()
    repo = InMemorySessionRepo()
    agent = Agent(stream_fn=reply_stream_fn("ok"))
    sess = await create_agent_session(agent, repo, ctx)
    sid = sess.session.metadata.id

    await agent.prompt("一")
    await agent.wait_for_idle()
    await agent.prompt("二")
    await agent.wait_for_idle()
    await sess.flush()
    await sess.dispose()

    agent2 = Agent(stream_fn=reply_stream_fn("ok"))
    sess2 = await create_agent_session(agent2, repo, ctx, session_id=sid)
    print("restored count=", len(agent2.state.messages))
    assert len(agent2.state.messages) == 4
    await sess2.dispose()


# ---------------------------------------------------------------------------
# laneConfig 恢复
# ---------------------------------------------------------------------------


async def test_laneConfig恢复system_prompt() -> None:
    ctx = Context()
    repo = InMemorySessionRepo()
    agent = Agent(stream_fn=reply_stream_fn("ok"))
    agent.set_system_prompt("你是专家")
    sess = await create_agent_session(agent, repo, ctx)
    sid = sess.session.metadata.id
    await sess.dispose()

    agent2 = Agent(stream_fn=reply_stream_fn("ok"))
    sess2 = await create_agent_session(agent2, repo, ctx, session_id=sid)
    print("restored prompt=", agent2.state.system_prompt)
    assert agent2.state.system_prompt == "你是专家"
    await sess2.dispose()


# ---------------------------------------------------------------------------
# 新建会话默认建 main 分支
# ---------------------------------------------------------------------------


async def test_新建会话使用默认分支() -> None:
    ctx = Context()
    repo = InMemorySessionRepo()
    agent = Agent(stream_fn=reply_stream_fn("ok"))
    sess = await create_agent_session(agent, repo, ctx)
    tip = await sess.session.get_branch_tip(DEFAULT_BRANCH, ctx)
    print("tip=", tip)
    assert tip is None  # 空分支，尚无 entry
    await sess.dispose()


# ---------------------------------------------------------------------------
# 零回归：不挂会话时 Agent 行为不变
# ---------------------------------------------------------------------------


async def test_不挂会话agent照常工作() -> None:
    agent = Agent(stream_fn=reply_stream_fn("独立"))
    await agent.prompt("hi")
    await agent.wait_for_idle()
    print("standalone messages=", len(agent.state.messages))
    assert len(agent.state.messages) == 2


async def test_dispose后不再落库() -> None:
    ctx = Context()
    repo = InMemorySessionRepo()
    agent = Agent(stream_fn=reply_stream_fn("ok"))
    sess = await create_agent_session(agent, repo, ctx)
    await sess.dispose()

    # dispose 后再 prompt，不应抛错也不应落库（订阅已取消）
    await agent.prompt("孤儿消息")
    await agent.wait_for_idle()
    # agent 自身状态照常更新
    assert len(agent.state.messages) == 2
