"""Full agent loop with REAL OpenAI streaming output (no API-free mock here).

Run:
    uv run python examples/openai_agent_streaming.py

What this shows end-to-end:
  1. user asks about the weather
  2. turn 1: assistant streams a tool_call (get_weather)
  3. the agent loop executes the real tool and feeds the ToolResultMessage back
  4. turn 2: assistant streams the final answer token-by-token ("typewriter")

Compared to the other examples:
  - openai_streaming.py : single provider call, no loop, no tools, sees raw deltas
  - agent_e2e.py        : full loop, but only prints at message_end (no typewriter)
  - this file           : full loop AND live text_delta rendering via message_update
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable, Mapping
from typing import Any

from dotenv import load_dotenv

from pi_agent.agent_core import (
    Agent,
    AgentEvent,
    AgentTool,
    AgentToolResult,
    AssistantMessage,
    Model,
    TextContent,
    ToolCall,
    ToolResultMessage,
)
from pi_agent.pi_ai import create_agent_stream_fn, create_default_registry

load_dotenv()


async def get_weather(
    tool_call_id: str,
    params: Mapping[str, Any],
    abort_event: asyncio.Event | None = None,
    on_update: Callable[[AgentToolResult[Any]], None] | None = None,
) -> AgentToolResult[Any]:
    """A real (deterministic, local) tool — no external weather API needed."""
    del tool_call_id, abort_event, on_update
    city = str(params.get("city", "Unknown"))
    return AgentToolResult(
        content=[TextContent(text=f"Sunny, 22C in {city}")],
        details={"city": city},
    )


def make_printer() -> Callable[[AgentEvent], None]:
    """Build an event listener that renders the loop like a chat UI.

    Streaming works because the loop wraps every provider delta event into a
    `message_update` agent event (see agent_loop._stream_assistant_response):
        event["assistant_message_event"] is the raw provider event,
        e.g. {"type": "text_delta", "delta": "Ship", ...}.
    """
    streamed_any_text = False  # did the current assistant message print any token?

    def on_event(event: AgentEvent) -> None:
        nonlocal streamed_any_text
        event_type = event["type"]

        if event_type == "message_start":
            message = event["message"]
            if isinstance(message, AssistantMessage):
                streamed_any_text = False
                print("[assistant] ", end="", flush=True)
            elif isinstance(message, ToolResultMessage):
                # tool results are already announced via tool_execution_*; stay quiet
                pass
            return

        if event_type == "message_update":
            inner = event.get("assistant_message_event") or {}
            inner_type = inner.get("type")

            if inner_type == "text_delta":
                print(inner["delta"], end="", flush=True)  # 打字机：逐 token
                streamed_any_text = True
            elif inner_type == "toolcall_end":
                tool_call = inner.get("tool_call")
                if isinstance(tool_call, ToolCall):
                    print(f"🔧 calling tool: {tool_call.name}({tool_call.arguments})")
            return

        if event_type == "tool_execution_start":
            print(f"[tool:start] {event['tool_name']} args={event['args']}")
            return

        if event_type == "tool_execution_end":
            print(f"[tool:end] {event['tool_name']} is_error={event['is_error']}")
            return

        if event_type == "message_end":
            message = event["message"]
            if isinstance(message, AssistantMessage):
                if streamed_any_text:
                    print()  # 收尾换行
                else:
                    # 纯工具调用回合（没有任何文本 token），把前缀行收掉
                    print("\r", end="")
            return

        if event_type == "turn_end":
            print("---")
            return

    return on_event


async def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Put it in .env and run "
            "`uv run python examples/openai_agent_streaming.py`."
        )

    model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-5-mini")
    base_url = os.getenv("OPENAI_API_BASE_URL", "")
    # responses       → OpenAI Responses API via the openai SDK (official OpenAI)
    # completions     → Chat Completions via the SDK (gateways with broken Responses SSE)
    # responses-sse   → Responses protocol via raw httpx SSE (Ark coding gateway: its
    #                   response.created event omits output[], crashing the SDK snapshot)
    style = os.getenv("OPENAI_API_STYLE", "responses").lower()

    if style == "completions":
        provider_name = "openai-completions"
        registry = create_default_registry()
    elif style == "responses-sse":
        from ark_responses_sse import build_ark_registry

        provider_name = "openai"  # adapter is registered under the standard key
        registry = build_ark_registry()
    else:
        provider_name = "openai"
        registry = create_default_registry()
    agent = Agent(
        stream_fn=create_agent_stream_fn(registry),
        session_id="openai-streaming-demo",
    )
    agent.set_model(
        Model(id=model_name, provider=provider_name, api=provider_name, base_url=base_url)
    )
    agent.set_system_prompt(
        "You are a concise assistant. Use get_weather when users ask weather "
        "questions. After getting the result, answer in one short sentence."
    )
    agent.set_tools(
        [
            AgentTool(
                name="get_weather",
                label="Get Weather",
                description="Returns a weather string for a city.",
                execute=get_weather,
                parameters={
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name, e.g. Paris"},
                    },
                    "required": ["city"],
                },
            )
        ]
    )
    agent.subscribe(make_printer())

    question = "What's the weather in Paris? Also, do I need an umbrella?"
    print(f"[user] {question}")
    await agent.prompt(question)

    final_message = agent.state.messages[-1]
    if isinstance(final_message, AssistantMessage):
        print(f"\nfinal stop_reason: {final_message.stop_reason}")
        if final_message.error_message:
            print(f"error: {final_message.error_message}")
        print(f"token usage: {final_message.usage.total_tokens}")


if __name__ == "__main__":
    asyncio.run(main())
