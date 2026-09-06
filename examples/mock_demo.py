"""Debug scaffold for MockProvider.

Run directly in PyCharm (right-click → Debug) or:
    uv run python examples/mock_demo.py

Covers the three branches of MockProvider._build_assistant_message:
  1. plain prompt         → Echo
  2. weather prompt       → toolUse (get_weather)
  3. tool result fed back → "Weather update: ..." summary

demo_agent_loop() then runs the real agent_loop with the mock model:
  mock emits toolUse → loop executes get_weather → result fed back automatically
  → mock's second turn emits the summary. No API key needed.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from typing import Any

from pi_agent.agent_core import (
    Agent,
    AgentEvent,
    AgentTool,
    AgentToolResult,
    AssistantMessage,
    LlmContext,
    Model,
    TextContent,
    ToolCall,
    ToolResultMessage,
)
from pi_agent.pi_ai import (
    PiAIRequest,
    ProviderRegistry,
    create_agent_stream_fn,
    create_default_registry,
    stream_simple,
)
from pi_agent.pi_ai.providers.mock import MockProvider


def _print_done(event: dict) -> None:
    message = event["message"]
    for block in message.content:
        if isinstance(block, TextContent):
            print(f"  text: {block.text!r}")
        elif isinstance(block, ToolCall):
            print(f"  tool_call: {block.name} args={block.arguments}")
    print(f"  stop_reason: {message.stop_reason}")


async def demo_echo(model: Model, registry: ProviderRegistry) -> None:
    """Branch 1: plain prompt → Echo."""
    print(">>> echo branch")
    stream = await stream_simple(
        "hello, who are you?",
        model=model,
        registry=registry,
    )
    async for event in stream:
        if event["type"] == "done":
            _print_done(event)


async def demo_tool_use(model: Model, registry: ProviderRegistry) -> ToolCall:
    """Branch 2: weather prompt → toolUse. Returns the emitted ToolCall."""
    print(">>> toolUse branch")
    stream = await stream_simple(
        "What's the weather in Paris?",
        model=model,
        registry=registry,
    )
    message = None
    async for event in stream:
        if event["type"] == "done":
            message = event["message"]
            _print_done(event)

    assert message is not None
    return next(block for block in message.content if isinstance(block, ToolCall))


async def demo_tool_result(tool_call: ToolCall) -> None:
    """Branch 3: feed a ToolResultMessage back → summary. Uses MockProvider
    directly, because stream_simple only accepts a prompt string."""
    print(">>> tool result branch")
    provider = MockProvider()
    request = PiAIRequest(
        model=Model(id="mock", provider="mock", api="mock"),
        context=LlmContext(
            system_prompt=None,
            messages=[
                ToolResultMessage(
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.name,
                    content=[TextContent(text="Sunny, 22C in Paris")],
                    is_error=False,
                    details={},
                ),
            ],
            tools=[],
        ),
    )
    stream = await provider.stream(request)
    async for event in stream:
        if event["type"] == "done":
            _print_done(event)


async def get_weather(
    tool_call_id: str,
    params: Mapping[str, Any],
    abort_event: asyncio.Event | None = None,
    on_update: Callable[[AgentToolResult[Any]], None] | None = None,
) -> AgentToolResult[Any]:
    """真工具：loop 在第二圈之前会执行它（签名对齐 ToolExecuteFn）。"""
    del tool_call_id, abort_event, on_update
    city = str(params.get("city", "Unknown"))
    return AgentToolResult(
        content=[TextContent(text=f"Sunny, 22C in {city}")],
        details={"city": city},
    )


async def demo_agent_loop() -> None:
    """完整 agent loop：mock 吐 toolUse → loop 真执行 get_weather →
    ToolResultMessage 自动回灌 → mock 第二圈吐 "Weather update"。
    和 demo_tool_result 的区别：这一圈不再由我们手动扮演。"""
    print(">>> agent loop (toolUse → real tool → summary)")

    def on_event(event: AgentEvent) -> None:
        event_type = event["type"]
        if event_type == "tool_execution_start":
            print(f"  [tool:start] {event['tool_name']} args={event['args']}")
        elif event_type == "tool_execution_end":
            print(f"  [tool:end] {event['tool_name']} is_error={event['is_error']}")
        elif event_type == "message_end" and isinstance(event["message"], AssistantMessage):
            for block in event["message"].content:
                if isinstance(block, TextContent):
                    print(f"  [assistant] text: {block.text!r}")
                elif isinstance(block, ToolCall):
                    print(f"  [assistant] tool_call: {block.name} args={block.arguments}")

    registry = create_default_registry()
    agent = Agent(stream_fn=create_agent_stream_fn(registry))
    agent.set_model(Model(id="mock", provider="mock", api="mock"))
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
    agent.subscribe(on_event)

    await agent.prompt("What's the weather in Paris?")

    final_message = agent.state.messages[-1]
    if isinstance(final_message, AssistantMessage):
        print(f"  final stop_reason: {final_message.stop_reason}")


async def main() -> None:
    registry = create_default_registry()
    model = Model(id="mock", provider="mock", api="mock")

    await demo_echo(model, registry)
    tool_call = await demo_tool_use(model, registry)
    await demo_tool_result(tool_call)
    await demo_agent_loop()


if __name__ == "__main__":
    asyncio.run(main())
