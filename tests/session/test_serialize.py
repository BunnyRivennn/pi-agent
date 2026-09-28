from __future__ import annotations

import pytest

from pi_agent.agent_core.types import (
    AssistantMessage,
    ImageContent,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
    Usage,
    UsageCost,
    UserMessage,
)
from pi_agent.session import (
    BranchSummaryEntry,
    CompactionEntry,
    CustomEntry,
    MessageEntry,
    NewEntry,
    SessionCorruptError,
    entry_from_dict,
    entry_to_dict,
    message_from_dict,
    message_to_dict,
)


def test_new_entry_omits_seq_and_timestamp() -> None:
    """NewEntry 语义：构造时不传 seq/timestamp，取哨兵默认（存储层后续分配）。"""
    entry: NewEntry = MessageEntry(
        id="a1",
        parent_id=None,
        message=UserMessage(content="hi", timestamp=1),
    )
    print("seq=", entry.seq, "timestamp=", entry.timestamp)
    assert entry.seq == -1
    assert entry.timestamp == 0
    # 仍可正常 roundtrip（存储层分配前也能序列化）
    out = entry_from_dict(entry_to_dict(entry))
    assert out == entry


def make_assistant() -> AssistantMessage:
    """构造一个完整的 AssistantMessage 用于测试。"""
    return AssistantMessage(
        content=[
            ThinkingContent(thinking="想一下", thinking_signature="sig-t"),
            TextContent(text="答案", text_signature="sig-x"),
            ToolCall(
                id="tc-1",
                name="read",
                arguments={"path": "a.py", "n": 3},
                thought_signature="sig-tc",
            ),
        ],
        api="mock-api",
        provider="mock-provider",
        model="mock-model",
        usage=Usage(
            input=10,
            output=20,
            cache_read=1,
            cache_write=2,
            total_tokens=33,
            cost=UsageCost(input=0.1, output=0.2, total=0.3),
        ),
        stop_reason="toolUse",
        timestamp=111,
    )


# ---------------------------------------------------------------------------
# 消息 roundtrip
# ---------------------------------------------------------------------------


def test_user_message_plain_text_roundtrip() -> None:
    """测试纯文本用户消息的序列化与反序列化往返。"""
    msg = UserMessage(content="你好", timestamp=1)
    out = message_from_dict(message_to_dict(msg))
    print("in=", msg, "out=", out)
    assert out == msg


def test_user_message_multimodal_block_roundtrip() -> None:
    """测试包含多模态内容块的用户消息的序列化与反序列化往返。"""
    msg = UserMessage(
        content=[
            TextContent(text="看图"),
            ImageContent(data="base64==", mime_type="image/png"),
        ],
        timestamp=2,
    )
    out = message_from_dict(message_to_dict(msg))
    print("out=", out)
    assert out == msg


def test_assistant_message_full_blocks_with_usage_roundtrip() -> None:
    """测试包含所有内容块和使用统计的助手消息的序列化与反序列化往返。"""
    msg = make_assistant()
    out = message_from_dict(message_to_dict(msg))
    print("out=", out)
    assert out == msg


def test_tool_result_message_roundtrip() -> None:
    """测试工具结果消息的序列化与反序列化往返。"""
    msg = ToolResultMessage(
        tool_call_id="tc-1",
        tool_name="read",
        content=[TextContent(text="file body")],
        is_error=False,
        details={"lines": 3},
        timestamp=5,
    )
    out = message_from_dict(message_to_dict(msg))
    print("out=", out)
    assert out == msg


# ---------------------------------------------------------------------------
# Entry roundtrip
# ---------------------------------------------------------------------------


def test_message_entry_roundtrip() -> None:
    """测试 MessageEntry 的序列化与反序列化往返。"""
    entry = MessageEntry(
        id="a1",
        parent_id=None,
        seq=0,
        timestamp=100,
        message=make_assistant(),
        terminate=True,
    )
    out = entry_from_dict(entry_to_dict(entry))
    print("out=", out)
    assert out == entry


def test_compaction_entry_with_retained_tail_roundtrip() -> None:
    """测试 CompactionEntry（带保留尾部消息）的序列化与反序列化往返。"""
    entry = CompactionEntry(
        id="c1",
        parent_id="a1",
        seq=1,
        timestamp=200,
        summary="## Goal\n做完 A0",
        retained_tail=[UserMessage(content="继续", timestamp=3), make_assistant()],
        tokens_before=1234,
        details={"read_files": ["a.py"]},
        usage=None,
        from_hook=False,
    )
    out = entry_from_dict(entry_to_dict(entry))
    print("out=", out)
    assert out == entry


def test_branch_summary_entry_roundtrip() -> None:
    """测试 BranchSummaryEntry 的序列化与反序列化往返。"""
    entry = BranchSummaryEntry(
        id="b1",
        parent_id="a1",
        seq=2,
        timestamp=300,
        from_id="x9",
        summary="放弃的分支摘要",
        from_hook=True,
    )
    out = entry_from_dict(entry_to_dict(entry))
    print("out=", out)
    assert out == entry


def test_custom_entry_roundtrip() -> None:
    """测试 CustomEntry 的序列化与反序列化往返。"""
    entry = CustomEntry(
        id="d1",
        parent_id="a1",
        seq=3,
        timestamp=400,
        custom_type="memory.note",
        data={"topic": "偏好", "value": "喜欢简洁"},
    )
    out = entry_from_dict(entry_to_dict(entry))
    print("out=", out)
    assert out == entry


# ---------------------------------------------------------------------------
# 显式失败：未知 type / 缺字段一律抛，不静默
# ---------------------------------------------------------------------------


def test_unknown_message_role_raises_error() -> None:
    """测试未知的消息角色会抛出 SessionCorruptError。"""
    with pytest.raises(SessionCorruptError):
        message_from_dict({"role": "hookMessage", "content": "x"})


def test_unknown_entry_type_raises_error() -> None:
    """测试未知的 entry 类型会抛出 SessionCorruptError。"""
    with pytest.raises(SessionCorruptError):
        entry_from_dict(
            {"id": "z", "parent_id": None, "seq": 0, "timestamp": 0, "type": "usage"}
        )


def test_unknown_content_block_raises_error() -> None:
    """测试未知的内容块类型会抛出 SessionCorruptError。"""
    with pytest.raises(SessionCorruptError):
        message_from_dict({"role": "user", "content": [{"type": "video"}]})


def test_missing_required_field_raises_error() -> None:
    """测试缺少必填字段会抛出 SessionCorruptError。"""
    with pytest.raises(SessionCorruptError):
        # assistant 缺 provider
        message_from_dict(
            {"role": "assistant", "content": [], "api": "a", "model": "m"}
        )
