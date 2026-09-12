"""容错工具参数 JSON 解析的测试。

覆盖三层：
1. parse_streaming_json 纯函数（严格 → 转义修复 → 截断补全 → 兜底 {}）
2. repair_json_string 字符级修复
3. 两个 provider 的集成：坏 JSON 参数在终态消息中被救回

对照 TS 版 pi 的 utils/json-parse.ts（parseStreamingJson 四级瀑布）。
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from typing import Any

import pytest

from pi_agent.agent_core import (
    LlmContext,
    Model,
    ToolCall,
    UserMessage,
)
from pi_agent.pi_ai import OpenAICompletionsProvider, OpenAIResponsesProvider, PiAIRequest
from pi_agent.pi_ai.providers._json_repair import (
    parse_streaming_json,
    repair_json_string,
)

# ============================================================================
# 1. parse_streaming_json
# ============================================================================


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # ① 第一级：严格解析即可
        ('{"city": "Shanghai"}', {"city": "Shanghai"}),
        ('{"n": 42, "ok": true}', {"n": 42, "ok": True}),
        # Mapping 直通，key 强制转 str
        ({"x": 1}, {"x": 1}),
        ({1: "a"}, {"1": "a"}),
        # 空值守卫
        ("", {}),
        ("   \n  ", {}),
        (None, {}),
        (123, {}),
        ([1, 2, 3], {}),
        # ② 第二级：字符串内脏字符的转义修复
        # 裸换行（真实 0x0A 出现在字符串值里，严格 json.loads 会拒绝）
        ('{"q": "a\nb"}', {"q": "a\nb"}),
        ('{"q": "a\tb"}', {"q": "a\tb"}),
        # 非法转义 \U（模型写 Windows 路径 C:\Users）→ 反斜杠翻倍救回；
        # 同一串里的 \n 是合法转义，解释为换行
        # \U 非法转义被翻倍为字面反斜杠；\n 合法转义解释为换行
        ('{"path": "C:\\Users\\new"}', {"path": "C:\\Users\new"}),
        # 结尾孤立反斜杠 → 翻倍
        ('{"p": "abc\\', {"p": "abc\\"}),
        # 合法的 \u 转义必须原样保留，不能被误改
        ('{"p": "a\\u0041b"}', {"p": "aAb"}),
        # ③ 第三级：截断结构补全
        ('{"city": "Shang', {"city": "Shang"}),
        ('{"a": 1,', {"a": 1}),
        ('{"a": [1, 2', {"a": [1, 2]}),
        ('{"a": 20', {"a": 20}),
        ('{"a": 1, "b": {', {"a": 1, "b": {}}),
        # 尾逗号 + 未闭合括号：去尾逗号、补括号即可救回（不删任何已生成内容）
        ('{"a": 1, "b": {"x": 2, ', {"a": 1, "b": {"x": 2}}),
        # 字符串值里的逗号不受影响，补引号收口
        ('{"q": "a, b', {"q": "a, b"}),
        # 救不回的残骸 → 必须显式降级 {}，绝不砍尾丢字段：
        # {"a": 1} 看似可救，但静默丢弃未生成完的 key 会让下游带着残缺参数执行；
        # {} 会被 jsonschema required 拦下，把错误喂回模型重试。
        ('{"a": 1, "cit', {}),
        ("{", {}),
        # ④ 垃圾输入：不抛异常，给 {}
        ("hello world", {}),
        ('{"a": 1}xxx', {}),  # 尾随垃圾必须拒绝，不能误解析
        # 顶层数组不是工具参数（必须是 object）→ {}
        ("[1, 2, 3", {}),
    ],
    ids=[
        "完整JSON",
        "数字和布尔类型",
        "dict直通",
        "dict的数字key转字符串",
        "空字符串",
        "纯空白",
        "None",
        "整数非str",
        "list非object",
        "裸换行",
        "裸Tab",
        "Windows路径非法转义",
        "结尾孤立反斜杠",
        "合法unicode转义保留",
        "截断-未闭合字符串",
        "截断-尾逗号",
        "截断-未闭合数组",
        "截断-数字半截",
        "截断-嵌套对象",
        "截断-嵌套尾逗号可补",
        "字符串内逗号不受影响",
        "残骸-半截key必须失败不可砍尾",
        "残骸-空对象",
        "垃圾文本",
        "尾随垃圾拒绝",
        "顶层数组不算参数",
    ],
)
def test_parse_streaming_json(raw: Any, expected: dict[str, Any]) -> None:
    result = parse_streaming_json(raw)
    # pytest -s 时可见：每个 case 喂了什么、救回了什么
    print(f"\n  输入: {raw!r}\n  输出: {result!r}\n  期望: {expected!r}")
    assert result == expected


def test_parse_streaming_json_never_raises() -> None:
    """任何奇形怪状的输入都不能抛异常（provider 在流式循环里调用它）。"""
    weird_inputs: list[Any] = [
        "{",
        "}",
        '"',
        "\\",
        "\x00\x01\x02",
        b'{"a": 1}',  # bytes 不是 str 也不是 Mapping
        object(),
        {"nested": {"deep": object()}},
    ]
    for raw in weird_inputs:
        result = parse_streaming_json(raw)
        print(f"\n  怪物输入: {raw!r:40} -> 安全降级: {result!r}")
        assert isinstance(result, dict)


def test_parser_never_silently_drops_fields() -> None:
    """防回归：解析器只修表达形式，绝不通过砍尾静默丢弃模型未吐完的字段。

    即使技术上能"救回"前面的完整字段，也必须返回 {} —— 显式失败让
    jsonschema required 校验拦下，由模型重试；部分参数被当成成功结果
    继续执行，才是真正危险的静默错误。
    """
    dangerous = [
        '{"a": 1, "cit',          # 半截 key
        '{"city": "Shanghai", "cou',  # 已完整的 city 也不能成为砍尾的理由
        '{"a": 1, "b": 2, "c":',  # 半截 key/value
    ]
    for raw in dangerous:
        print(f"\n  禁止砍尾: {raw!r} -> {parse_streaming_json(raw)!r}（必须是 {{}}）")
        assert parse_streaming_json(raw) == {}


# ============================================================================
# 2. repair_json_string
# ============================================================================


def test_repair_doubles_illegal_escape() -> None:
    # \U 不是合法 JSON 转义 → 反斜杠翻倍；\n 合法 → 保留
    before, after = '{"p": "C:\\Users"}', repair_json_string('{"p": "C:\\Users"}')
    print(f"\n  修非法转义: {before} -> {after}")
    assert after == '{"p": "C:\\\\Users"}'


def test_repair_escapes_raw_control_characters() -> None:
    # 字符串值里的真实换行会被转义成 \n（两字符），让 JSON 重新合法
    before, after = '{"q": "a\nb"}', repair_json_string('{"q": "a\nb"}')
    print(f"\n  修裸换行  : {before!r} -> {after!r}")
    assert after == '{"q": "a\\nb"}'


def test_repair_handles_trailing_backslash() -> None:
    # 孤立反斜杠翻倍为两个；repair 只修字符，不负责补引号（那是补全级的事）
    before, after = '{"p": "abc\\', repair_json_string('{"p": "abc\\')
    print(f"\n  修尾反斜杠: {before!r} -> {after!r}")
    assert after == '{"p": "abc\\\\'


def test_repair_leaves_valid_json_unchanged() -> None:
    valid = '{"a": 1, "b": "x\\ny\\t", "c": "\\u0041"}'
    after = repair_json_string(valid)
    print(f"\n  合法串不动: {valid} -> {after}")
    assert after == valid


# ============================================================================
# 3. 集成：OpenAI Responses provider
# ============================================================================


@pytest.mark.asyncio
async def test_responses_provider_repairs_illegal_escape_in_tool_arguments() -> None:
    # 模型把 Windows 路径直接塞进参数：\U 是非法 JSON 转义，
    # 旧实现 json.loads 失败 → arguments 整体变 {}；容错解析应原样救回。
    broken_args = '{"path": "C:\\Users"}'
    expected_args = {"path": "C:\\Users"}

    async def stream_request_fn(
        _payload: dict[str, Any],
        _api_key: str,
        _base_url: str | None,
    ) -> AsyncIterator[Mapping[str, Any]]:
        async def _events() -> AsyncIterator[Mapping[str, Any]]:
            yield {
                "type": "response.output_item.added",
                "item": {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "read_file",
                },
            }
            yield {
                "type": "response.function_call_arguments.done",
                "call_id": "call_1",
                "name": "read_file",
                "arguments": broken_args,
            }
            yield {
                "type": "response.completed",
                "response": {
                    "status": "completed",
                    "output": [
                        {
                            "type": "function_call",
                            "call_id": "call_1",
                            "name": "read_file",
                            "arguments": broken_args,
                        }
                    ],
                },
            }

        return _events()

    provider = OpenAIResponsesProvider(stream_request_fn=stream_request_fn)
    request = PiAIRequest(
        model=Model(id="test", provider="openai", api="openai"),
        context=LlmContext(messages=[UserMessage(content="read it")]),
        api_key="test-key",
    )

    stream = await provider.stream(request)
    toolcall_end_args: list[dict[str, Any]] = []
    async for event in stream:
        if event["type"] == "toolcall_end":
            toolcall_end_args.append(dict(event["tool_call"].arguments))
    message = await stream.result()

    final_args = [b.arguments for b in message.content if isinstance(b, ToolCall)]
    print(f"\n  Responses 集成: 坏参数 {broken_args!r}")
    print(f"  toolcall_end 事件参数: {toolcall_end_args}")
    print(f"  终态消息参数        : {final_args}")
    assert toolcall_end_args == [expected_args]
    tool_calls = [block for block in message.content if isinstance(block, ToolCall)]
    assert tool_calls[0].arguments == expected_args


# ============================================================================
# 4. 集成：OpenAI Completions provider
# ============================================================================


@pytest.mark.asyncio
async def test_completions_provider_completes_truncated_streaming_arguments() -> None:
    # 流在参数吐到一半时结束（无完整 JSON，只有一个 finish_reason）：
    # _finish_current_block 走 buffer 兜底解析，应补全救回已生成的字段。
    async def stream_request_fn(
        _payload: dict[str, Any],
        _api_key: str,
        _base_url: str | None,
    ) -> AsyncIterator[Mapping[str, Any]]:
        async def _events() -> AsyncIterator[Mapping[str, Any]]:
            yield {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "index": 0,
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"city": "Shang',
                                    },
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ]
            }
            yield {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}

        return _events()

    provider = OpenAICompletionsProvider(stream_request_fn=stream_request_fn)
    request = PiAIRequest(
        model=Model(id="test", provider="openai", api="openai-completions"),
        context=LlmContext(messages=[UserMessage(content="weather?")]),
        api_key="test-key",
    )

    stream = await provider.stream(request)
    async for _ in stream:
        pass
    message = await stream.result()

    tool_calls = [block for block in message.content if isinstance(block, ToolCall)]
    print('\n  Completions 流式截断: 碎片 \'{"city": "Shang\' 流就结束')
    print(f"  终态救回参数: {tool_calls[0].arguments!r}，stop_reason={message.stop_reason}")
    assert message.stop_reason == "toolUse"
    assert tool_calls[0].arguments == {"city": "Shang"}


@pytest.mark.asyncio
async def test_completions_provider_repairs_illegal_escape_non_streaming() -> None:
    # 非流式响应：message.tool_calls.function.arguments 是坏 JSON 字符串
    broken_args = '{"path": "C:\\Users"}'

    async def request_fn(
        _payload: dict[str, Any],
        _api_key: str,
        _base_url: str | None,
    ) -> Mapping[str, Any]:
        return {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "read_file",
                                    "arguments": broken_args,
                                },
                            }
                        ],
                    },
                }
            ],
        }

    provider = OpenAICompletionsProvider(request_fn=request_fn)
    request = PiAIRequest(
        model=Model(id="test", provider="openai", api="openai-completions"),
        context=LlmContext(messages=[UserMessage(content="read it")]),
        api_key="test-key",
    )

    stream = await provider.stream(request)
    async for _ in stream:
        pass
    message = await stream.result()

    tool_calls = [block for block in message.content if isinstance(block, ToolCall)]
    print(f"\n  Completions 非流式: 坏参数 {broken_args!r}")
    print(f"  终态救回参数: {tool_calls[0].arguments!r}")
    assert tool_calls[0].arguments == {"path": "C:\\Users"}
