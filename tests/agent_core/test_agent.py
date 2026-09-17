from __future__ import annotations

import asyncio

import pytest

from pi_agent.agent_core import (
    Agent,
    AgentLoopConfig,
    AssistantMessageEventStream,
    LlmContext,
    Model,
    TextContent,
)
from pi_agent.agent_core.types import AssistantMessage, StopReason, Usage


def make_assistant(text: str, *, stop_reason: StopReason = "stop") -> AssistantMessage:
    return AssistantMessage(
        content=[TextContent(text=text)],
        api="mock-api",
        provider="mock-provider",
        model="mock-model",
        stop_reason=stop_reason,
        usage=Usage(),
    )


def blocking_stream(
    _model: Model,
    _context: LlmContext,
    _config: AgentLoopConfig,
    _abort_event: asyncio.Event | None,
) -> AssistantMessageEventStream:
    stream = AssistantMessageEventStream()

    async def _emit() -> None:
        await asyncio.sleep(0)
        stream.push({"type": "start", "partial": make_assistant("")})

    asyncio.create_task(_emit())
    return stream


@pytest.mark.asyncio
async def test_agent_has_sane_default_state() -> None:
    agent = Agent(stream_fn=lambda *_: AssistantMessageEventStream())

    assert agent.state.system_prompt == ""
    assert agent.state.model is not None
    assert agent.state.thinking_level == "off"
    assert agent.state.tools == []
    assert agent.state.messages == []
    assert agent.state.is_streaming is False


@pytest.mark.asyncio
async def test_prompt_while_streaming_raises() -> None:
    def stream_fn(
        _model: Model,
        _context: LlmContext,
        _config: AgentLoopConfig,
        abort_event: asyncio.Event | None,
    ) -> AssistantMessageEventStream:
        stream = AssistantMessageEventStream()

        async def _emit() -> None:
            await asyncio.sleep(0)
            stream.push({"type": "start", "partial": make_assistant("")})
            while abort_event is not None and not abort_event.is_set():
                await asyncio.sleep(0.01)
            stream.push(
                {
                    "type": "error",
                    "reason": "aborted",
                    "error": make_assistant("", stop_reason="aborted"),
                }
            )

        asyncio.create_task(_emit())
        return stream

    agent = Agent(stream_fn=stream_fn)

    first = asyncio.create_task(agent.prompt("first"))
    await asyncio.sleep(0.02)

    with pytest.raises(RuntimeError, match="already processing a prompt"):
        await agent.prompt("second")

    agent.abort()
    await first


@pytest.mark.asyncio
async def test_continue_while_streaming_raises() -> None:
    def stream_fn(
        _model: Model,
        _context: LlmContext,
        _config: AgentLoopConfig,
        abort_event: asyncio.Event | None,
    ) -> AssistantMessageEventStream:
        stream = AssistantMessageEventStream()

        async def _emit() -> None:
            await asyncio.sleep(0)
            stream.push({"type": "start", "partial": make_assistant("")})
            while abort_event is not None and not abort_event.is_set():
                await asyncio.sleep(0.01)
            stream.push(
                {
                    "type": "error",
                    "reason": "aborted",
                    "error": make_assistant("", stop_reason="aborted"),
                }
            )

        asyncio.create_task(_emit())
        return stream

    agent = Agent(stream_fn=stream_fn)

    first = asyncio.create_task(agent.prompt("first"))
    await asyncio.sleep(0.02)

    with pytest.raises(RuntimeError, match="already processing"):
        await agent.continue_()

    agent.abort()
    await first


@pytest.mark.asyncio
async def test_continue_from_assistant_message_raises() -> None:
    def stream_fn(
        _model: Model,
        _context: LlmContext,
        _config: AgentLoopConfig,
        _abort_event: asyncio.Event | None,
    ) -> AssistantMessageEventStream:
        return AssistantMessageEventStream()

    agent = Agent(stream_fn=stream_fn)
    agent.append_message(make_assistant("done"))

    with pytest.raises(RuntimeError, match="Cannot continue from message role"):
        await agent.continue_()


@pytest.mark.asyncio
async def test_agent_forwards_session_id_to_stream_fn() -> None:
    received_session_ids: list[str | None] = []

    def stream_fn(
        _model: Model,
        _context: LlmContext,
        config: AgentLoopConfig,
        _abort_event: asyncio.Event | None,
    ) -> AssistantMessageEventStream:
        received_session_ids.append(config.session_id)
        stream = AssistantMessageEventStream()

        async def _emit() -> None:
            await asyncio.sleep(0)
            stream.push({"type": "done", "reason": "stop", "message": make_assistant("ok")})

        asyncio.create_task(_emit())
        return stream

    agent = Agent(stream_fn=stream_fn, session_id="session-1")
    agent.set_model(Model(id="mock-model", provider="mock-provider", api="mock-api"))

    await agent.prompt("hello")

    agent.session_id = "session-2"
    await agent.prompt("again")

    assert received_session_ids == ["session-1", "session-2"]


# ⭐ 练习：listener 抛异常不能炸死 agent（由你来填完 4 个空）
@pytest.mark.asyncio
async def test_listener_exception_does_not_kill_agent() -> None:
    """订阅者只是旁观者：它的 bug 不能影响 agent 运行，也不能连累其他订阅者。

    修复前：bad_listener raise → 炸穿 _execute 的消费循环 → agent 走 except 分支，
            历史里被追加一条合成 error 消息，prompt 行为异常。
    修复后：agent 正常跑完，good_listener 照常收到全部事件。
    """

    # 一个永远正常完成一轮对话的假模型（照抄 test_agent_forwards_session_id_to_stream_fn）
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
                {"type": "done", "reason": "stop", "message": make_assistant("hi")}
            )

        asyncio.create_task(_emit())
        return stream

    agent = Agent(stream_fn=stream_fn)

    received: list[str] = []  # 好 listener 用它留痕：记录自己收到的事件类型

    def bad_listener(event: dict) -> None:
        # ⬜ 填空 1：这个 listener 是个"有 bug 的订阅者"，让它直接抛异常
        # 提示：一行 raise，异常文本写 "listener boom"
        raise NotImplementedError("填空 1：在这里 raise RuntimeError")

    def good_listener(event: dict) -> None:
        received.append(str(event["type"]))

    # ⬜ 填空 2：把两个 listener 都注册到 agent 上（调用 agent.subscribe(...)）
    # 提示：两行；注意注册顺序——让 bad 排在 good 前面，才能证明"前者炸了后者还会被调用"
    raise NotImplementedError("填空 2：在这里 subscribe 两个 listener")

    # ⬜ 填空 3（When）：像现有测试一样对 agent 说一句话并等它跑完
    # 提示：一行 await。修复前这里会被 bad_listener 的异常波及；修复后正常返回
    raise NotImplementedError("填空 3：在这里 await agent.prompt(...)")

    # ⬜ 填空 4（Then）：把下面三条断言补全
    # 4a. agent 跑完后不应该还处于 streaming 状态
    #     提示：assert agent.state.is_streaming is ...
    raise NotImplementedError("填空 4a：断言 is_streaming")

    # 4b. 好 listener 必须确实收到了事件（坏 listener 没连累它）
    #     提示：至少会收到 "agent_start"；断言 "agent_start" 在 received 里，
    #           并且 "agent_end" 也在（agent 完整跑到了结束）
    raise NotImplementedError("填空 4b：断言 good_listener 收到了 agent_start/agent_end")

    # 4c. agent 的对话历史必须是"正常的那一轮"，而不是错误兜底：
    #     最后一条消息是普通 AssistantMessage（"hi"），且没有 error_message
    #     提示：last = agent.state.messages[-1]
    #           assert isinstance(last, AssistantMessage)
    #           assert last.error_message is None
    raise NotImplementedError("填空 4c：断言历史末条是正常 assistant 消息，无 error_message")
