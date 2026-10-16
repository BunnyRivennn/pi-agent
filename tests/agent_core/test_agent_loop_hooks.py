"""循环级 hook 插入点测试：挂上中间件，走 agent.prompt() 真的会触发。

和 ``tests/agent_core/test_hooks.py`` 的分工：那边测注册表本身（聚合语义），
这边测插入点接上了没——前者全绿而后者全红，正是「写好了但循环里没调」的样子。
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from pi_agent.agent_core import (
    Agent,
    AgentLoopConfig,
    AssistantMessageEventStream,
    LlmContext,
    Model,
    TextContent,
    UserMessage,
)
from pi_agent.agent_core.hooks import HookRegistry
from pi_agent.agent_core.types import AssistantMessage, Usage


def _assistant(text: str) -> AssistantMessage:
    return AssistantMessage(
        content=[TextContent(text=text)],
        api="mock-api",
        provider="mock-provider",
        model="mock-model",
        stop_reason="stop",
        usage=Usage(),
    )


def _agent(hooks: HookRegistry, seen_contexts: list[list[Any]] | None = None) -> Agent:
    """一个每次都正常收尾的假模型。"""

    def stream_fn(
        _model: Model,
        context: LlmContext,
        _config: AgentLoopConfig,
        _abort_event: asyncio.Event | None,
    ) -> AssistantMessageEventStream:
        if seen_contexts is not None:
            seen_contexts.append(list(context.messages))
        stream = AssistantMessageEventStream()

        async def _emit() -> None:
            await asyncio.sleep(0)
            stream.push({"type": "done", "reason": "stop", "message": _assistant("hi")})

        asyncio.create_task(_emit())
        return stream

    return Agent(stream_fn=stream_fn, hooks=hooks)


@pytest.mark.asyncio
async def test_before_run注入的消息进了本轮上下文() -> None:
    hooks = HookRegistry()

    async def inject(_event: dict[str, Any], _abort: Any = None) -> dict[str, Any]:
        return {"messages": [UserMessage(content="<profile/>")]}

    hooks.on("before_run", inject)

    seen: list[list[Any]] = []
    agent = _agent(hooks, seen)
    await agent.prompt("你好")
    await agent.wait_for_idle()

    # 这轮 LLM 必须看见注入的那条，且排在用户输入之后
    texts = [str(getattr(m, "content", "")) for m in seen[0]]
    assert "你好" in texts[0]
    assert any("<profile/>" in t for t in texts)


@pytest.mark.asyncio
async def test_before_run注入的消息也落进了历史() -> None:
    """只进上下文不进历史的话，恢复会话时两边就对不上了。"""
    hooks = HookRegistry()

    async def inject(_event: dict[str, Any], _abort: Any = None) -> dict[str, Any]:
        return {"messages": [UserMessage(content="<profile/>")]}

    hooks.on("before_run", inject)
    agent = _agent(hooks)
    await agent.prompt("你好")
    await agent.wait_for_idle()

    texts = [str(getattr(m, "content", "")) for m in agent.state.messages]
    assert any("<profile/>" in t for t in texts)


@pytest.mark.asyncio
async def test_没挂before_run时一切照旧() -> None:
    hooks = HookRegistry()
    seen: list[list[Any]] = []
    agent = _agent(hooks, seen)
    await agent.prompt("你好")
    await agent.wait_for_idle()

    assert len(seen[0]) == 1  # 只有用户那一条


@pytest.mark.asyncio
async def test_before_run返回pending的assistant消息被拦住() -> None:
    hooks = HookRegistry()

    async def inject(_event: dict[str, Any], _abort: Any = None) -> dict[str, Any]:
        pending = _assistant("半截")
        pending.stop_reason = "pending"
        return {"messages": [pending]}

    hooks.on("before_run", inject)
    agent = _agent(hooks)
    await agent.prompt("你好")
    await agent.wait_for_idle()

    # 带着 pending 跑下去，后续流程会以为"还有没收尾的回复"；
    # 循环的兜底把这个异常转成一条 error 消息，不会静默吞掉
    errors = [
        getattr(m, "error_message", None)
        for m in agent.state.messages
        if getattr(m, "stop_reason", None) == "error"
    ]
    assert errors and "pending assistant message" in str(errors[0])


@pytest.mark.asyncio
async def test_before_run_end拿到本轮完整消息且只触发一次() -> None:
    hooks = HookRegistry()
    calls: list[list[Any]] = []

    async def observe(event: dict[str, Any], _abort: Any = None) -> None:
        calls.append(list(event["messages"]))
        return None

    hooks.on("before_run_end", observe)
    agent = _agent(hooks)
    await agent.prompt("你好")
    await agent.wait_for_idle()

    assert len(calls) == 1
    roles = [getattr(m, "role", None) for m in calls[0]]
    assert roles == ["user", "assistant"]


@pytest.mark.asyncio
async def test_before_run_end的followup让agent多跑一轮() -> None:
    hooks = HookRegistry()
    fired = 0

    async def follow(_event: dict[str, Any], _abort: Any = None) -> dict[str, Any] | None:
        nonlocal fired
        fired += 1
        if fired == 1:
            return {"follow_up": "再说一点"}
        return None

    hooks.on("before_run_end", follow)
    agent = _agent(hooks)
    await agent.prompt("你好")
    await agent.wait_for_idle()

    texts = [str(getattr(m, "content", "")) for m in agent.state.messages]
    assert any("再说一点" in t for t in texts)
    assert fired == 2  # 追加的那轮跑完又问了一次，这次没活了
