"""会话持久化层的异常类型。"""

from __future__ import annotations


class SessionError(Exception):
    """会话层错误基类。"""


class SessionCorruptError(SessionError):
    """序列化/反序列化遇到未知或损坏的数据时抛出。

    显式失败优于静默丢字段：残缺 payload 若被静默接受，下游可能拿着不完整的
    历史继续运行。携带可定位的上下文（如 entry id / 字段名）便于诊断。
    """


class CompactionRaceError(SessionError):
    """乐观压缩提交时 branch tip 已被推进（有新消息落库）。

    压缩需先在屏障外扫路径、算切点、调 LLM 生成摘要（慢 IO），期间可能有新消息
    append 进来。提交时若发现 tip 已变，放弃本次压缩（下一轮末重试），而非把过时
    的摘要接到错误的父节点上。这是可预期的良性竞态，调用方应静默跳过。
    """


__all__ = ["SessionError", "SessionCorruptError", "CompactionRaceError"]
