from __future__ import annotations

import pytest

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


def test_content_text_string_direct() -> None:
    assert content_text("hello") == "hello"


def test_content_text_concatenate_text_ignore_non_text() -> None:
    blocks = [TextContent(text="a"), ThinkingContent(thinking="x"), TextContent(text="b")]
    print("joined=", content_text(blocks))
    assert content_text(blocks) == "ab"


def test_content_text_empty_uses_default() -> None:
    assert content_text([], default="(empty)") == "(empty)"


# ---------------------------------------------------------------------------
# estimate_tokens：CJK 感知（有意偏离官方 chars/4）
# ---------------------------------------------------------------------------


def test_estimate_tokens_english_by_quarter() -> None:
    # 8 个 ASCII 字符 → ceil(8/4) = 2
    msg = UserMessage(content="abcdefgh", timestamp=0)
    print("tokens=", estimate_tokens(msg))
    assert estimate_tokens(msg) == 2


def test_estimate_tokens_chinese_one_to_one() -> None:
    # 4 个汉字 → 4 token（官方 chars/4 会算成 1，这是我们的偏离）
    msg = UserMessage(content="你好世界", timestamp=0)
    print("cjk tokens=", estimate_tokens(msg))
    assert estimate_tokens(msg) == 4


def test_estimate_tokens_mixed_count_separately() -> None:
    # 2 汉字(=2) + 4 ASCII(ceil(4/4)=1) = 3
    msg = UserMessage(content="中文abcd", timestamp=0)
    print("mixed tokens=", estimate_tokens(msg))
    assert estimate_tokens(msg) == 3


def test_estimate_tokens_assistant_with_toolcall() -> None:
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


def test_calculate_context_tokens_prefer_total() -> None:
    assert calculate_context_tokens(Usage(total_tokens=999, input=1)) == 999


def test_calculate_context_tokens_no_total_sum() -> None:
    u = Usage(input=10, output=20, cache_read=5, cache_write=5)
    assert calculate_context_tokens(u) == 40


def test_estimate_context_no_usage_all_estimated() -> None:
    msgs = [UserMessage(content="abcdefgh", timestamp=0)]  # 2 tokens
    est = estimate_context_tokens(msgs)
    print("est=", est)
    assert est.last_usage_index is None
    assert est.tokens == 2


def test_estimate_context_with_usage_use_usage_plus_tail() -> None:
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


def test_estimate_context_skip_error_usage() -> None:
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


def test_should_compact_exceeds_threshold_triggers() -> None:
    s = CompactionSettings(enabled=True, reserve_tokens=10000, keep_recent_tokens=20000)
    assert should_compact(95000, 100000, s) is True
    assert should_compact(89000, 100000, s) is False


def test_should_compact_disabled_never_triggers() -> None:
    s = CompactionSettings(enabled=False, reserve_tokens=10000, keep_recent_tokens=20000)
    assert should_compact(95000, 100000, s) is False


def test_default_settings_values_match_official() -> None:
    assert DEFAULT_COMPACTION_SETTINGS.reserve_tokens == 16384
    assert DEFAULT_COMPACTION_SETTINGS.keep_recent_tokens == 20000
    assert DEFAULT_COMPACTION_SETTINGS.enabled is True


def test_should_compact_non_positive_threshold_raises() -> None:
    """小窗口 + 字段默认值：旧行为是恒为真且压不动（空转），现在直接报。"""
    s = CompactionSettings()  # reserve=16384，按 128k 窗口定的
    with pytest.raises(ValueError, match="压缩阈值非正"):
        should_compact(5000, 8000, s)


def test_should_compact_disabled_does_not_check_threshold() -> None:
    """enabled=False 优先于阈值校验：显式关掉就不该因配置报错。"""
    s = CompactionSettings(enabled=False)
    assert should_compact(5000, 8000, s) is False


# ---------------------------------------------------------------------------
# CompactionSettings.for_window
# ---------------------------------------------------------------------------


def test_for_window_scales_proportionally() -> None:
    s = CompactionSettings.for_window(128_000)
    assert s.reserve_tokens == 16_000  # 1/8
    assert s.keep_recent_tokens == 32_000  # 1/4
    assert s.enabled is True


def test_for_window_small_window_no_longer_spin() -> None:
    """这是 main.py 里那个补丁要解决的问题，现在由库保证。"""
    s = CompactionSettings.for_window(8000)
    assert s.reserve_tokens == 1000
    assert s.keep_recent_tokens == 2000
    # 阈值 = 8000 - 1000 = 7000，为正，不再恒为真
    assert should_compact(7500, 8000, s) is True
    assert should_compact(6500, 8000, s) is False
    # 保留额度小于窗口，切点能攒满
    assert s.keep_recent_tokens < 8000


def test_for_window_minimum_window_has_lower_bound() -> None:
    s = CompactionSettings.for_window(100)
    assert s.reserve_tokens == 256
    assert s.keep_recent_tokens == 256


def test_for_window_can_be_disabled() -> None:
    assert CompactionSettings.for_window(128_000, enabled=False).enabled is False


def test_for_window_rejects_non_positive_window() -> None:
    with pytest.raises(ValueError, match="必须为正"):
        CompactionSettings.for_window(0)


# ---------------------------------------------------------------------------
# serialize_conversation
# ---------------------------------------------------------------------------


def test_serialize_each_role_section() -> None:
    msgs: list[AgentMessage] = [
        UserMessage(content="问题", timestamp=0),
        assistant(
            ThinkingContent(thinking="想一下"),
            TextContent(text="回答"),
            ToolCall(id="1", name="run", arguments={"a": 1}),
        ),
        ToolResultMessage(
            tool_call_id="1",
            tool_name="run",
            content=[TextContent(text="结果")],
            is_error=False,
        ),
    ]
    out = serialize_conversation(msgs)
    print("serialized=\n", out)
    assert "[User]: 问题" in out
    assert "[Assistant thinking]: 想一下" in out
    assert "[Assistant]: 回答" in out
    assert "[Assistant tool calls]: run(a=1)" in out
    assert "[Tool result]: 结果" in out


def test_serialize_toolresult_truncated_if_too_long() -> None:
    long_text = "x" * (TOOL_RESULT_MAX_CHARS + 500)
    msgs = [
        ToolResultMessage(
            tool_call_id="1",
            tool_name="run",
            content=[TextContent(text=long_text)],
            is_error=False,
        ),
    ]
    out = serialize_conversation(msgs)
    print("truncated tail=", out[-60:])
    assert "more characters truncated" in out
    assert len(out) < len(long_text)


def test_serialize_empty_messages_empty_string() -> None:
    assert serialize_conversation([]) == ""
