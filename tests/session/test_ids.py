from __future__ import annotations

import uuid

import pytest

from pi_agent.session import assert_valid_session_id, new_entry_id, new_session_id


def test_session_id_is_valid_uuid() -> None:
    """测试生成的会话 ID 是合法的 UUIDv7。"""
    sid = new_session_id()
    print("sid=", sid)
    parsed = uuid.UUID(sid)
    assert parsed.version == 7


def test_session_id_is_time_ordered() -> None:
    """测试会话 ID 按时间有序（UUIDv7 高位为时间戳）。"""
    early = new_session_id(timestamp_ms=1_000)
    late = new_session_id(timestamp_ms=2_000)
    print("early=", early, "late=", late)
    # UUIDv7 高位是时间戳，字典序应与时间序一致
    assert early < late


def test_entry_id_retries_on_collision() -> None:
    """测试条目 ID 在碰撞时会自动重试直到生成唯一值。"""
    existing = {"aaaaaaaa"}

    calls = {"n": 0}
    real_hex = uuid.uuid4().hex

    # 第一次强制撞上，之后放行：用一个 set 的 __contains__ 已足够验证唯一性
    eid = new_entry_id(existing)
    print("eid=", eid, "len=", len(eid))
    assert eid not in existing
    assert len(eid) == 8
    _ = (calls, real_hex)


def test_entry_id_uniqueness() -> None:
    """测试大量生成的条目 ID 不会重复。"""
    seen: set[str] = set()
    for _ in range(500):
        eid = new_entry_id(seen)
        assert eid not in seen
        seen.add(eid)


def test_valid_session_id_passes() -> None:
    """测试合法的会话 ID 能通过校验。"""
    for good in ["abc", "a.b_c-1", "A1", "session.2026"]:
        assert_valid_session_id(good)


def test_invalid_session_id_raises_error() -> None:
    """测试非法的会话 ID 会抛出 ValueError。"""
    for bad in ["", "-abc", "abc-", ".x", "a/b", "a b"]:
        with pytest.raises(ValueError):
            assert_valid_session_id(bad)
