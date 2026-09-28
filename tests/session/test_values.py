from __future__ import annotations

import pytest

from pi_agent.session import (
    Value,
    ValueList,
    Write,
    append_list,
    branch_tip,
    delete_value,
    lane_config,
    list_,
    session_name,
    set_value,
    value,
)
from pi_agent.session.values import (
    ListAppendWrite,
    ValueDeleteWrite,
    ValueSetWrite,
)


def test_write_combines_and_exports_covers_value_and_list_writes() -> None:
    """测试写入操作可以组合并导出，涵盖值和列表写入。"""
    writes: list[Write] = [
        set_value(value("pi.x", "k"), 1),
        delete_value(value("pi.x", "k")),
        append_list(list_("pi.log"), "line"),
    ]
    print("writes=", writes)
    assert len(writes) == 3


def test_value_address_construction() -> None:
    """测试值地址的构造，包含命名空间和键。"""
    a: Value[str] = value("ns", "k")
    print("addr=", a)
    assert a.namespace == "ns"
    assert a.key == "k"
    assert a.kind == "value"


def test_list_address_construction() -> None:
    """测试列表地址的构造，仅包含命名空间。"""
    a: ValueList[str] = list_("ns")
    assert a.key == ""
    assert a.kind == "list"


def test_empty_namespace_raises_error() -> None:
    """测试空命名空间会抛出 ValueError。"""
    with pytest.raises(ValueError):
        value("")


def test_namespace_with_nul_raises_error() -> None:
    """测试命名空间包含 NUL 字符会抛出 ValueError。"""
    with pytest.raises(ValueError):
        value("a\u0000b")


def test_set_value_constructs_write() -> None:
    """测试 set_value 构造 ValueSetWrite，包含正确的操作符和值。"""
    w = set_value(value("pi.x", "k"), {"a": 1})
    print("w=", w)
    assert isinstance(w, ValueSetWrite)
    assert w.op == "set"
    assert w.value == {"a": 1}


def test_delete_value_constructs_write() -> None:
    """测试 delete_value 构造 ValueDeleteWrite，包含正确的操作符。"""
    w = delete_value(value("pi.x", "k"))
    assert isinstance(w, ValueDeleteWrite)
    assert w.op == "delete"


def test_append_list_constructs_write() -> None:
    """测试 append_list 构造 ListAppendWrite，包含正确的操作符和值。"""
    w = append_list(list_("pi.log"), "line")
    assert isinstance(w, ListAppendWrite)
    assert w.op == "append"
    assert w.value == "line"


def test_named_addresses() -> None:
    """测试命名地址辅助函数返回正确的命名空间和键。"""
    assert lane_config("main").namespace == "pi.lane.config"
    assert lane_config("main").key == "main"
    assert branch_tip("main").namespace == "pi.branch.tip"
    assert session_name().namespace == "pi.session.name"
    assert session_name().key == ""
