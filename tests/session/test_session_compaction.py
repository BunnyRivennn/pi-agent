"""D2 切点选择纯函数测试（session/compaction.py）。

重点锁定关键不变量：切点绝不落在 toolResult 上（否则拆散 toolCall/toolResult 对）。
"""

from __future__ import annotations

from pi_agent.agent_core.types import (
    AssistantMessage,
    TextContent,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from pi_agent.session.compaction import (
    collect_retained_tail,
    find_cut_point,
    find_turn_start_index,
    find_valid_cut_points,
)
from pi_agent.session.types import CompactionEntry, Entry, MessageEntry


def _user(text: str) -> UserMessage:
    return UserMessage(content=text, timestamp=0)


def _assistant_tool(name: str, call_id: str) -> AssistantMessage:
    return AssistantMessage(
        content=[TextContent(text="calling"), ToolCall(id=call_id, name=name, arguments={})],
        api="a",
        provider="p",
        model="m",
        stop_reason="toolUse",
        usage=Usage(),
    )


def _tool_result(call_id: str, name: str) -> ToolResultMessage:
    return ToolResultMessage(
        tool_call_id=call_id,
        tool_name=name,
        content=[TextContent(text="result")],
        is_error=False,
    )


def _msg_entry(entry_id: str, parent: str | None, seq: int, msg: object) -> MessageEntry:
    return MessageEntry(id=entry_id, parent_id=parent, seq=seq, message=msg)  # type: ignore[arg-type]


def _conversation() -> list[Entry]:
    # user -> assistant(toolUse) -> toolResult -> assistant(text) -> user ...
    return [
        _msg_entry("e0", None, 1, _user("q1")),
        _msg_entry("e1", "e0", 2, _assistant_tool("run", "c1")),
        _msg_entry("e2", "e1", 3, _tool_result("c1", "run")),
        _msg_entry(
            "e3", "e2", 4,
            AssistantMessage(
                content=[TextContent(text="a1")], api="a", provider="p",
                model="m", stop_reason="stop", usage=Usage(),
            ),
        ),
        _msg_entry("e4", "e3", 5, _user("q2")),
    ]


# ---------------------------------------------------------------------------
# find_valid_cut_points
# ---------------------------------------------------------------------------


def test_valid_cut_points_跳过toolresult() -> None:
    entries = _conversation()
    cuts = find_valid_cut_points(entries, 0, len(entries))
    print("cuts=", cuts)
    # e2 是 toolResult，不能作切点
    assert 2 not in cuts
    # user/assistant 都可切
    assert set(cuts) == {0, 1, 3, 4}


def test_valid_cut_points_空范围() -> None:
    assert find_valid_cut_points(_conversation(), 2, 2) == []


def test_valid_cut_points_含compaction不可切() -> None:
    entries: list[Entry] = [
        CompactionEntry(id="c", parent_id=None, seq=1, summary="s"),
        _msg_entry("e1", "c", 2, _user("q")),
    ]
    cuts = find_valid_cut_points(entries, 0, len(entries))
    # compaction 不是有效切点；只有 e1
    assert cuts == [1]


# ---------------------------------------------------------------------------
# find_turn_start_index
# ---------------------------------------------------------------------------


def test_turn_start_回溯到user() -> None:
    entries = _conversation()
    # e2(toolResult) 所属 turn 的起点是 e0(user)
    assert find_turn_start_index(entries, 2, 0) == 0


def test_turn_start_找不到返回负一() -> None:
    entries: list[Entry] = [
        _msg_entry("e0", None, 1, _assistant_tool("run", "c1")),
        _msg_entry("e1", "e0", 2, _tool_result("c1", "run")),
    ]
    assert find_turn_start_index(entries, 1, 0) == -1


# ---------------------------------------------------------------------------
# find_cut_point
# ---------------------------------------------------------------------------


def test_cut_point_切点不落在toolresult() -> None:
    entries = _conversation()
    # 小 keep_recent 迫使切点尽量靠后，但仍必须避开 toolResult
    result = find_cut_point(entries, 0, len(entries), keep_recent_tokens=1)
    print("cut=", result)
    assert entries[result.first_kept_entry_index].type == "message"
    kept_msg = entries[result.first_kept_entry_index].message  # type: ignore[union-attr]
    assert getattr(kept_msg, "role", None) in ("user", "assistant")


def test_cut_point_大阈值保留全部() -> None:
    entries = _conversation()
    result = find_cut_point(entries, 0, len(entries), keep_recent_tokens=10_000)
    # 阈值远大于总量 → 切点在最前，等于不压缩
    assert result.first_kept_entry_index == 0


def test_cut_point_空返回起点() -> None:
    result = find_cut_point([], 0, 0, keep_recent_tokens=100)
    assert result.first_kept_entry_index == 0


# ---------------------------------------------------------------------------
# collect_retained_tail
# ---------------------------------------------------------------------------


def test_retained_tail_收集切点之后的消息() -> None:
    entries = _conversation()
    tail = collect_retained_tail(entries, 3, len(entries))
    # e3(assistant) + e4(user)
    assert len(tail) == 2
    assert isinstance(tail[0], AssistantMessage)
    assert isinstance(tail[1], UserMessage)


def test_retained_tail_起点即末尾为空() -> None:
    entries = _conversation()
    assert collect_retained_tail(entries, len(entries), len(entries)) == []
