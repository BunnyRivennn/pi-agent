from __future__ import annotations

from pi_agent.agent_core.compaction import (
    DEFAULT_COMPACTION_SETTINGS,
    TOOL_RESULT_MAX_CHARS,
    CompactionSettings,
    calculate_context_tokens,
    content_text,
    estimate_context_tokens,
    estimate_tokens,
    serialize_conversation,
    should_compact,
)
from pi_agent.agent_core.types import (
    AgentMessage,
    AssistantMessage,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)


def assistant(
    *blocks: object,
    stop_reason: str = "stop",
    usage: Usage | None = None,
) -> AssistantMessage:
    return AssistantMessage(
        content=list(blocks),  # type: ignore[arg-type]
        api="a",
        provider="p",
        model="m",
        stop_reason=stop_reason,  # type: ignore[arg-type]
        usage=usage or Usage(),
    )


# ---------------------------------------------------------------------------
# content_text
# ---------------------------------------------------------------------------


def test_content_text_字符串直返() -> None:
    assert content_text("hello") == "hello"


def test_content_text_拼接文本块忽略非文本() -> None:
    blocks = [TextContent(text="a"), ThinkingContent(thinking="x"), TextContent(text="b")]
    print("joined=", content_text(blocks))
    assert content_text(blocks) == "ab"


def test_content_text_空则返回default() -> None:
    assert content_text([], default="(empty)") == "(empty)"


# ---------------------------------------------------------------------------
# estimate_tokens：CJK 感知（有意偏离官方 chars/4）
# ---------------------------------------------------------------------------


def test_estimate_tokens_英文按四分之一() -> None:
    # 8 个 ASCII 字符 → ceil(8/4) = 2
    msg = UserMessage(content="abcdefgh", timestamp=0)
    print("tokens=", estimate_tokens(msg))
    assert estimate_tokens(msg) == 2


def test_estimate_tokens_中文按一比一() -> None:
    # 4 个汉字 → 4 token（官方 chars/4 会算成 1，这是我们的偏离）
    msg = UserMessage(content="你好世界", timestamp=0)
    print("cjk tokens=", estimate_tokens(msg))
    assert estimate_tokens(msg) == 4


def test_estimate_tokens_中英混合分别计() -> None:
    # 2 汉字(=2) + 4 ASCII(ceil(4/4)=1) = 3
    msg = UserMessage(content="中文abcd", timestamp=0)
    print("mixed tokens=", estimate_tokens(msg))
    assert estimate_tokens(msg) == 3


def test_estimate_tokens_assistant含toolcall() -> None:
    msg = assistant(
        TextContent(text="ok"),
        ToolCall(id="1", name="run", arguments={"x": 1}),
    )
    # 有值即可，重点是不抛异常且 > 0
    print("assistant tokens=", estimate_tokens(msg))
    assert estimate_tokens(msg) > 0


def test_estimate_tokens_toolresult() -> None:
    msg = ToolResultMessage(
        tool_call_id="1",
        tool_name="run",
        content=[TextContent(text="abcdefgh")],
        is_error=False,
    )
    assert estimate_tokens(msg) == 2


# ---------------------------------------------------------------------------
# calculate_context_tokens / estimate_context_tokens
# ---------------------------------------------------------------------------


def test_calculate_context_tokens_优先total() -> None:
    assert calculate_context_tokens(Usage(total_tokens=999, input=1)) == 999


def test_calculate_context_tokens_无total则求和() -> None:
    u = Usage(input=10, output=20, cache_read=5, cache_write=5)
    assert calculate_context_tokens(u) == 40


def test_estimate_context_无usage全估算() -> None:
    msgs = [UserMessage(content="abcdefgh", timestamp=0)]  # 2 tokens
    est = estimate_context_tokens(msgs)
    print("est=", est)
    assert est.last_usage_index is None
    assert est.tokens == 2


def test_estimate_context_有usage则用usage加尾部() -> None:
    msgs: list[AgentMessage] = [
        UserMessage(content="hi", timestamp=0),
        assistant(TextContent(text="x"), usage=Usage(total_tokens=1000)),
        UserMessage(content="abcdefgh", timestamp=0),  # 尾部 2 tokens
    ]
    est = estimate_context_tokens(msgs)
    print("est=", est)
    assert est.last_usage_index == 1
    assert est.usage_tokens == 1000
    assert est.trailing_tokens == 2
    assert est.tokens == 1002


def test_estimate_context_跳过error的usage() -> None:
    msgs: list[AgentMessage] = [
        assistant(TextContent(text="x"), stop_reason="error", usage=Usage(total_tokens=9999)),
        UserMessage(content="abcdefgh", timestamp=0),
    ]
    est = estimate_context_tokens(msgs)
    # error 的 usage 不算，退回全估算
    print("est=", est)
    assert est.last_usage_index is None


# ---------------------------------------------------------------------------
# should_compact
# ---------------------------------------------------------------------------


def test_should_compact_超阈值触发() -> None:
    s = CompactionSettings(enabled=True, reserve_tokens=10000, keep_recent_tokens=20000)
    assert should_compact(95000, 100000, s) is True
    assert should_compact(89000, 100000, s) is False


def test_should_compact_关闭时永不触发() -> None:
    s = CompactionSettings(enabled=False, reserve_tokens=10000, keep_recent_tokens=20000)
    assert should_compact(95000, 100000, s) is False


def test_default_settings_值对齐官方() -> None:
    assert DEFAULT_COMPACTION_SETTINGS.reserve_tokens == 16384
    assert DEFAULT_COMPACTION_SETTINGS.keep_recent_tokens == 20000
    assert DEFAULT_COMPACTION_SETTINGS.enabled is True


# ---------------------------------------------------------------------------
# serialize_conversation
# ---------------------------------------------------------------------------


def test_serialize_各角色分段() -> None:
    msgs: list[AgentMessage] = [
        UserMessage(content="问题", timestamp=0),
        assistant(
            ThinkingContent(thinking="想一下"),
            TextContent(text="回答"),
            ToolCall(id="1", name="run", arguments={"a": 1}),
        ),
        ToolResultMessage(
            tool_call_id="1", tool_name="run",
            content=[TextContent(text="结果")], is_error=False,
        ),
    ]
    out = serialize_conversation(msgs)
    print("serialized=\n", out)
    assert "[User]: 问题" in out
    assert "[Assistant thinking]: 想一下" in out
    assert "[Assistant]: 回答" in out
    assert "[Assistant tool calls]: run(a=1)" in out
    assert "[Tool result]: 结果" in out


def test_serialize_toolresult超长截断() -> None:
    long_text = "x" * (TOOL_RESULT_MAX_CHARS + 500)
    msgs = [
        ToolResultMessage(
            tool_call_id="1", tool_name="run",
            content=[TextContent(text=long_text)], is_error=False,
        ),
    ]
    out = serialize_conversation(msgs)
    print("truncated tail=", out[-60:])
    assert "more characters truncated" in out
    assert len(out) < len(long_text)


def test_serialize_空消息产出空串() -> None:
    assert serialize_conversation([]) == ""
