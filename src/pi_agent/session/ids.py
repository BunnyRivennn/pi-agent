"""ID 生成（对齐官方 session-manager.ts 的 createSessionId / generateId）。

- session id：UUIDv7（时间有序），标准库无内置，按 RFC 9562 手工构造。
- entry id：8 位 hex，带碰撞检查（对齐官方 generateId：最多重试 100 次，仍冲突则退回完整 UUID）。
"""

from __future__ import annotations

import os
import re
import time
import uuid
from typing import Protocol

# session id 合法性：非空、仅字母数字与 '-' '_' '.'、首尾为字母数字。
_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?$")


class SupportsHas(Protocol):
    def __contains__(self, key: str) -> bool: ...


def new_session_id(timestamp_ms: int | None = None) -> str:
    """生成 UUIDv7 字符串（时间有序，128 bit，RFC 9562）。

    结构：48 bit 毫秒时间戳 | 4 bit version(7) | 12 bit rand_a
          | 2 bit variant(0b10) | 62 bit rand_b
    """
    ms = timestamp_ms if timestamp_ms is not None else int(time.time() * 1000)
    ms &= (1 << 48) - 1
    rand = int.from_bytes(os.urandom(10), "big")  # 80 随机 bit
    rand_a = (rand >> 62) & 0xFFF  # 12 bit
    rand_b = rand & ((1 << 62) - 1)  # 62 bit

    value = ms << 80
    value |= 0x7 << 76  # version 7
    value |= rand_a << 64
    value |= 0b10 << 62  # variant
    value |= rand_b
    return str(uuid.UUID(int=value))


def assert_valid_session_id(session_id: str) -> None:
    """校验外部传入的 session id 合法（对齐官方 assertValidSessionId）。"""
    if not _SESSION_ID_RE.match(session_id):
        raise ValueError(
            "Session id must be non-empty, contain only alphanumeric characters, "
            "'-', '_', and '.', and start and end with an alphanumeric character"
        )


def new_entry_id(existing: SupportsHas) -> str:
    """生成唯一的 8 位 hex entry id；与 ``existing`` 冲突则重试。

    对齐官方 generateId：最多 100 次，仍冲突则退回完整 UUID（hex）。
    ``existing`` 只需支持 ``in`` 判断（如已有 id 的 set / dict）。
    """
    for _ in range(100):
        candidate = uuid.uuid4().hex[:8]
        if candidate not in existing:
            return candidate
    return uuid.uuid4().hex


__all__ = [
    "new_session_id",
    "assert_valid_session_id",
    "new_entry_id",
    "SupportsHas",
]
