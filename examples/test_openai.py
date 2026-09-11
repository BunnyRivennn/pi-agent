# debug_stream.py
import asyncio
import json
from collections.abc import AsyncIterator, Mapping

# 假设这个文件叫 openai_responses_provider.py
# 你把要调试的文件放在同目录下
from .. import (
    _consume_openai_event_stream,
    _new_partial_message,
    _apply_openai_stream_event,
    _OpenAIStreamingState,
    AssistantMessageEventStream,
)


# 模拟一个简单的 Model
class FakeModel:
    def __init__(self):
        self.id = "gpt-4o-mini"
        self.api = "openai"
        self.provider = "openai"


# 模拟 OpenAI 返回的原始事件流
async def fake_openai_events() -> AsyncIterator[Mapping[str, any]]:
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
    yield {
        "type": "response.output_item.added",
        "output_index": 1,
        "item": {
            "type": "function_call",
            "id": "call_123",
            "call_id": "call_123",
            "name": "get_weather",
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
                {"type": "function_call", "call_id": "call_123", "name": "get_weather",
                 "arguments": {"city": "Shanghai"}},
            ],
            "usage": {"input_tokens": 50, "output_tokens": 100, "total_tokens": 150},
        },
    }


async def main():
    model = FakeModel()
    stream = AssistantMessageEventStream()

    print("=" * 60)
    print("开始消费 OpenAI 事件流...")
    print("=" * 60)

    # 消费事件流
    await _consume_openai_event_stream(
        stream=stream,
        model=model,
        events=fake_openai_events(),
    )

    print("\n" + "=" * 60)
    print("事件流消费完毕，以下是所有事件：")
    print("=" * 60)

    # 查看所有事件
    for i, event in enumerate(stream.events):
        event_type = event.get("type", "unknown")
        print(f"\n[{i}] type: {event_type}")

        if event_type == "text_delta":
            print(f"    delta: '{event.get('delta', '')}'")
        elif event_type == "text_end":
            print(f"    text: '{event.get('content', '')}'")
        elif event_type == "toolcall_start":
            print(f"    call_id: {event.get('partial', {}).content[event['content_index']].id}")
            print(f"    name: {event.get('partial', {}).content[event['content_index']].name}")
        elif event_type == "toolcall_delta":
            print(f"    delta: '{event.get('delta', '')}'")
        elif event_type == "toolcall_end":
            tc = event.get("tool_call", {})
            print(f"    call_id: {tc.id}, name: {tc.name}, args: {tc.arguments}")
        elif event_type == "done":
            msg = event.get("message", {})
            print(f"    reason: {event.get('reason', '')}")
            print(f"    stop_reason: {msg.stop_reason}")
            for block in msg.content:
                if hasattr(block, 'text'):
                    print(f"    text: '{block.text}'")
                elif hasattr(block, 'name'):
                    print(f"    tool_call: {block.name}({block.arguments})")
        elif event_type == "start":
            print(f"    partial 已创建")
        else:
            print(f"    {event}")


if __name__ == "__main__":
    asyncio.run(main())
