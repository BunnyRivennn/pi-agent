"""会话层压缩：切点选择 + compact 编排（D2）。

对齐官方 harness ``compaction.ts`` 中操作 session ``Entry`` 的部分：
- ``find_valid_cut_points`` / ``find_cut_point``：在 branch 路径上选一个安全切点，
  切点之后的消息作为 ``retained_tail`` 内联保留，之前的被摘要取代。
- ``build_compaction_entry``：把序列化 + 摘要文本组装成待写入的 ``CompactionEntry``。

切点安全规则（关键）：**切点不能落在 toolResult 上**。否则会把 toolCall 与其
toolResult 拆散——被切走的 assistant 发起了工具调用，保留的 toolResult 却没有对应
的调用，模型会拒绝或困惑。因此有效切点只在 user/assistant 边界。

token 估算复用 agent_core.compaction（含 CJK 感知偏离）。摘要 LLM 调用不在本模块，
由调用方（AgentSession）注入，保持本层纯粹、可测。
"""

from __future__ import annotations

from ..agent_core.compaction import estimate_tokens
from ..agent_core.types import AgentMessage
from .types import CompactionEntry, Entry, MessageEntry


def _entry_message(entry: Entry) -> AgentMessage | None:
    if isinstance(entry, MessageEntry):
        return entry.message
    return None


def find_valid_cut_points(
    entries: list[Entry], start_index: int, end_index: int
) -> list[int]:
    """返回 [start_index, end_index) 内可作为切点的下标。

    只有 message 型 entry 且其消息角色为 user/assistant 才是有效切点；
    toolResult 不可切（会拆散 toolCall/toolResult 对）；branch_summary 也可切。
    """
    cut_points: list[int] = []
    for i in range(start_index, end_index):
        entry = entries[i]
        if entry.type == "message":
            msg = _entry_message(entry)
            role = getattr(msg, "role", None)
            if role in ("user", "assistant"):
                cut_points.append(i)
        elif entry.type == "branch_summary":
            cut_points.append(i)
    return cut_points


def find_turn_start_index(
    entries: list[Entry], entry_index: int, start_index: int
) -> int:
    """从 entry_index 往回找包含它的那个 turn 的起点（user 消息或 branch_summary）。

    找不到返回 -1。用于判断切点是否切进了一个进行中的 turn（split-turn）。
    """
    for i in range(entry_index, start_index - 1, -1):
        entry = entries[i]
        if entry.type == "branch_summary":
            return i
        if entry.type == "message":
            msg = _entry_message(entry)
            if getattr(msg, "role", None) == "user":
                return i
    return -1


class CutPointResult:
    """切点选择结果。"""

    __slots__ = ("first_kept_entry_index", "turn_start_index", "is_split_turn")

    def __init__(
        self,
        first_kept_entry_index: int,
        turn_start_index: int,
        is_split_turn: bool,
    ) -> None:
        self.first_kept_entry_index = first_kept_entry_index
        self.turn_start_index = turn_start_index
        self.is_split_turn = is_split_turn

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return (
            f"CutPointResult(first_kept_entry_index={self.first_kept_entry_index}, "
            f"turn_start_index={self.turn_start_index}, "
            f"is_split_turn={self.is_split_turn})"
        )


def find_cut_point(
    entries: list[Entry],
    start_index: int,
    end_index: int,
    keep_recent_tokens: int,
) -> CutPointResult:
    """选一个切点，使切点之后大致保留 ``keep_recent_tokens`` 的近期上下文。

    从后往前累加消息 token，累计到阈值时，取该位置之后第一个有效切点。
    然后把切点向前回退，跳过非 message 条目，直到紧邻一条 message 或 compaction。
    """
    cut_points = find_valid_cut_points(entries, start_index, end_index)
    if not cut_points:
        return CutPointResult(start_index, -1, False)

    accumulated = 0
    cut_index = cut_points[0]
    for i in range(end_index - 1, start_index - 1, -1):
        entry = entries[i]
        if entry.type != "message":
            continue
        msg = _entry_message(entry)
        if msg is not None:
            accumulated += estimate_tokens(msg)
        if accumulated >= keep_recent_tokens:
            for cp in cut_points:
                if cp >= i:
                    cut_index = cp
                    break
            break

    # 回退：切点不落在悬空的非 message 条目后面。
    while cut_index > start_index:
        prev = entries[cut_index - 1]
        if prev.type == "compaction":
            break
        if prev.type == "message":
            break
        cut_index -= 1

    cut_entry = entries[cut_index]
    cut_msg = _entry_message(cut_entry)
    is_user = (
        cut_entry.type == "message" and getattr(cut_msg, "role", None) == "user"
    )
    turn_start = -1 if is_user else find_turn_start_index(entries, cut_index, start_index)
    return CutPointResult(
        first_kept_entry_index=cut_index,
        turn_start_index=turn_start,
        is_split_turn=(not is_user and turn_start != -1),
    )


def collect_retained_tail(
    entries: list[Entry], first_kept_entry_index: int, end_index: int
) -> list[AgentMessage]:
    """收集切点及其之后的 message 条目的消息，作为 CompactionEntry.retained_tail。"""
    tail: list[AgentMessage] = []
    for i in range(first_kept_entry_index, end_index):
        msg = _entry_message(entries[i])
        if msg is not None:
            tail.append(msg)
    return tail


def build_compaction_entry(
    *,
    entry_id: str,
    parent_id: str | None,
    summary: str,
    retained_tail: list[AgentMessage],
    tokens_before: int,
    usage: dict[str, object] | None = None,
    from_hook: bool = False,
) -> CompactionEntry:
    """组装一个待写入的 CompactionEntry（seq/timestamp 由存储层分配）。"""
    return CompactionEntry(
        id=entry_id,
        parent_id=parent_id,
        summary=summary,
        retained_tail=retained_tail,
        tokens_before=tokens_before,
        usage=usage,
        from_hook=from_hook,
    )


__all__ = [
    "CutPointResult",
    "find_valid_cut_points",
    "find_turn_start_index",
    "find_cut_point",
    "collect_retained_tail",
    "build_compaction_entry",
]
