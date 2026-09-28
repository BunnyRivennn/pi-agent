"""会话持久化演示：退出 → 重启 → 续聊（免 API key）。

运行：
    uv run python examples/session_demo.py

演示内容：
  1. 第一个"进程"：新建落盘会话，跑一轮带工具的对话，消息自动落库。
  2. dispose() 关闭会话（模拟进程退出）。
  3. 第二个"进程"：用同一 session_id 重开，历史消息自动灌回 Agent。
  4. 在恢复的上下文上继续对话，新消息追加到同一分支。

db 文件落在 <临时目录>/pi-agent-demo-sessions/<session-id>.db，
用 SqliteSessionRepo 管理——这就是"每会话一个 db 文件"的落点。
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from collections.abc import Callable, Mapping
from typing import Any

from pi_agent.agent_core import (
    Agent,
    AgentTool,
    AgentToolResult,
    AssistantMessage,
    Model,
    TextContent,
    ToolCall,
)
from pi_agent.agent_core.types import ToolResultMessage, UserMessage
from pi_agent.pi_ai import create_agent_stream_fn, create_default_registry
from pi_agent.session import (
    DEFAULT_BRANCH,
    Context,
    SqliteSessionRepo,
    create_agent_session,
)

MOCK_MODEL = Model(id="mock", provider="mock", api="mock")


async def get_weather(
    tool_call_id: str,
    params: Mapping[str, Any],
    abort_event: asyncio.Event | None = None,
    on_update: Callable[[AgentToolResult[Any]], None] | None = None,
) -> AgentToolResult[Any]:
    del tool_call_id, abort_event, on_update
    city = str(params.get("city", "Unknown"))
    return AgentToolResult(
        content=[TextContent(text=f"Sunny, 22C in {city}")],
        details={"city": city},
    )


def make_agent() -> Agent:
    """每次"进程启动"都造一个全新的 Agent（无任何内存残留）。"""
    agent = Agent(stream_fn=create_agent_stream_fn(create_default_registry()))
    agent.set_model(MOCK_MODEL)
    agent.set_system_prompt("你是一个天气助手。")
    agent.set_tools(
        [
            AgentTool(
                name="get_weather",
                label="Get Weather",
                description="Returns a weather string for a city.",
                execute=get_weather,
            )
        ]
    )
    return agent


def describe(message: object, index: int) -> str:
    if isinstance(message, UserMessage):
        content = message.content
        text = content if isinstance(content, str) else _blocks_to_text(content)
        return f"  [{index}] user: {text!r}"
    if isinstance(message, AssistantMessage):
        parts: list[str] = []
        for block in message.content:
            if isinstance(block, TextContent):
                parts.append(repr(block.text))
            elif isinstance(block, ToolCall):
                parts.append(f"tool_call({block.name} {block.arguments})")
        return f"  [{index}] assistant: {' + '.join(parts)}"
    if isinstance(message, ToolResultMessage):
        return f"  [{index}] toolResult: {message.tool_name}"
    return f"  [{index}] {type(message).__name__}"


def _blocks_to_text(blocks: object) -> str:
    if not isinstance(blocks, list):
        return str(blocks)
    texts = [b.text for b in blocks if isinstance(b, TextContent)]
    return " ".join(texts)


def dump(agent: Agent, title: str) -> None:
    print(f"  {title}（共 {len(agent.state.messages)} 条）")
    for index, message in enumerate(agent.state.messages):
        print(describe(message, index))


async def main() -> None:
    root = os.path.join(tempfile.gettempdir(), "pi-agent-demo-sessions")
    ctx = Context()
    repo = SqliteSessionRepo(root)
    print(f"会话根目录: {root}\n")

    # ---------------- 第一个"进程" ----------------
    print("=== 进程 1：新建会话，跑一轮带工具的对话 ===")
    agent1 = make_agent()
    sess1 = await create_agent_session(agent1, repo, ctx)
    session_id = sess1.session.metadata.id
    print(f"session_id = {session_id}")
    print(f"db 文件   = {os.path.join(root, session_id + '.db')}")

    await agent1.prompt("What's the weather in Paris?")
    await agent1.wait_for_idle()
    await sess1.flush()  # 等落库任务写完

    dump(agent1, "进程 1 结束时的上下文")
    stats = await sess1.session.get_stats(ctx)
    print(f"  已落库 message 条目: {stats.message_count}")

    await sess1.dispose()  # 模拟进程退出
    print("  -> 会话已关闭（进程退出）\n")

    # ---------------- 第二个"进程" ----------------
    print("=== 进程 2：用同一 session_id 重开，历史自动恢复 ===")
    agent2 = make_agent()
    print(f"  重开前 agent2 消息数: {len(agent2.state.messages)}（全新 Agent）")

    sess2 = await create_agent_session(agent2, repo, ctx, session_id=session_id)
    dump(agent2, "恢复后的上下文")
    print(f"  恢复的 system_prompt: {agent2.state.system_prompt!r}")

    # ---------------- 在恢复的上下文上续聊 ----------------
    print("\n=== 进程 2：在恢复的历史上继续对话 ===")
    await agent2.prompt("那伦敦呢？")
    await agent2.wait_for_idle()
    await sess2.flush()

    dump(agent2, "续聊后的上下文")
    stats2 = await sess2.session.get_stats(ctx)
    print(f"  累计落库 message 条目: {stats2.message_count}")

    # 分支 tip 指向最新一条 entry
    tip = await sess2.session.get_branch_tip(DEFAULT_BRANCH, ctx)
    print(f"  分支 {DEFAULT_BRANCH} 的 tip entry: {tip}")

    await sess2.dispose()

    # ---------------- 会话列表 ----------------
    ids = await repo.list_ids(ctx)
    print(f"\n仓库中的会话: {ids}")
    print("（再次运行本示例会新建一个会话；db 文件保留在上面的根目录）")


if __name__ == "__main__":
    asyncio.run(main())
