# debug_provider_events.py
# 不调真实 API：手工喂一串伪造的 OpenAI Responses SSE 事件，
# 单步调试 pi-agent 的 provider 是怎么把原始事件翻译成框架事件的。
#
# 右键 Debug 运行即可；断点可以打在：
#   src/pi_agent/pi_ai/providers/openai.py
#     - _consume_openai_event_stream（消费循环入口）
#     - _apply_openai_stream_event（单个事件分发器）
#     - _ensure_tool_call / _emit_tool_end_if_needed（工具调用生命周期）
import asyncio
from collections.abc import AsyncIterator, Mapping

from pi_agent.agent_core import Model
from pi_agent.agent_core.event_stream import AssistantMessageEventStream
from pi_agent.pi_ai.providers.openai import (
    _OpenAIStreamingState,  # noqa: F401 （调试时在变量面板里能看到这个类型）
    _consume_openai_event_stream,
)


# 模拟一个 Model（用真的 Model dataclass，别手写 FakeModel：
# provider 内部还会读 model.api / model.provider / model.base_url）
model = Model(id="gpt-4o-mini", provider="openai", api="openai")


# 模拟 OpenAI 返回的原始事件流
async def fake_openai_events() -> AsyncIterator[Mapping[str, object]]:
    """模拟 OpenAI Responses API 的 SSE 事件"""

    # 1. 先输出一段文本
    yield {"type": "response.output_text.delta", "delta": "巴", "output_index": 0}
    yield {"type": "response.output_text.delta", "delta": "黎", "output_index": 0}
    yield {"type": "response.output_text.delta", "delta": "今", "output_index": 0}
    yield {"type": "response.output_text.delta", "delta": "天", "output_index": 0}
    yield {"type": "response.output_text.delta", "delta": " ", "output_index": 0}
    yield {"type": "response.output_text.delta", "delta": "2", "output_index": 0}
    yield {"type": "response.output_text.delta", "delta": "0", "output_index": 0}
    yield {"type": "response.output_text.delta", "delta": "度", "output_index": 0}
    yield {"type": "response.output_text.done", "text": "巴黎今天20度", "output_index": 0}

    # 2. 再调一个工具
    #    注意：真实 DeepSeek 这里 id 是 UUID、call_id 是 call_00_...，
    #    而下面 delta 事件的 item_id 用的是 UUID —— 两个 id 不一致，
    #    provider 靠 tool_item_id_to_call_id 映射表把它们对上。
    #    这里为了简单让两者相同；想复现 DeepSeek 场景可把 id 改成别的值试试。
    yield {
        "type": "response.output_item.added",
        "output_index": 1,
        "item": {
            "type": "function_call",
            "id": "call_123",
            "call_id": "call_123",
            "name": "get_weather",
            "arguments": "",
        },
    }
    yield {
        "type": "response.function_call_arguments.delta",
        "item_id": "call_123",
        "delta": '{"cit',
    }
    yield {
        "type": "response.function_call_arguments.delta",
        "item_id": "call_123",
        "delta": 'y": "Shanghai"}',
    }
    yield {
        "type": "response.function_call_arguments.done",
        "item_id": "call_123",
        "arguments": '{"city": "Shanghai"}',
    }

    # 3. 完成
    yield {
        "type": "response.completed",
        "response": {
            "output": [
                {"type": "message", "content": [{"type": "output_text", "text": "巴黎今天20度"}]},
                {
                    "type": "function_call",
                    "call_id": "call_123",
                    "name": "get_weather",
                    "arguments": {"city": "Shanghai"},
                },
            ],
            "usage": {"input_tokens": 50, "output_tokens": 100, "total_tokens": 150},
        },
    }


async def main():
    stream = AssistantMessageEventStream()

    print("=" * 60)
    print("开始消费 OpenAI 事件流...")
    print("=" * 60)

    # 生产者：把伪造事件喂进翻译器，翻译出的框架事件进 queue
    await _consume_openai_event_stream(
        stream=stream,
        model=model,
        events=fake_openai_events(),
    )

    print("\n" + "=" * 60)
    print("事件流消费完毕，以下是所有事件：")
    print("=" * 60)

    # 消费者：EventStream 是 asyncio.Queue，不是 list，
    # 必须 async for 逐个取（done 事件后队列里有哨兵，循环自动结束）
    i = 0
    async for event in stream:
        event_type = event.get("type", "unknown")
        print(f"\n[{i}] type: {event_type}")

        if event_type == "text_delta":
            print(f"    delta: '{event.get('delta', '')}'")
        elif event_type == "text_end":
            print(f"    text: '{event.get('content', '')}'")
        elif event_type == "toolcall_start":
            block = event["partial"].content[event["content_index"]]
            print(f"    call_id: {block.id}")
            print(f"    name: {block.name}")
        elif event_type == "toolcall_delta":
            print(f"    delta: '{event.get('delta', '')}'")
        elif event_type == "toolcall_end":
            tc = event.get("tool_call")
            print(f"    call_id: {tc.id}, name: {tc.name}, args: {tc.arguments}")
        elif event_type == "done":
            msg = event.get("message")
            print(f"    reason: {event.get('reason', '')}")
            print(f"    stop_reason: {msg.stop_reason}")
            for block in msg.content:
                if hasattr(block, "text"):
                    print(f"    text: '{block.text}'")
                elif hasattr(block, "name"):
                    print(f"    tool_call: {block.name}({block.arguments})")
        elif event_type == "start":
            print("    partial 已创建")
        else:
            print(f"    {event}")

        i += 1


if __name__ == "__main__":
    asyncio.run(main())
