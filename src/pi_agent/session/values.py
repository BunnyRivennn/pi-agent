"""KV + List 侧存储的类型化地址与写操作（对齐官方 values.ts）。

侧存储保存"非 entry"的会话状态：模型/思考等级配置、会话名、标签、分支 tip 等。
- ``value(namespace, key)``  -> 单值地址 Value[T]
- ``list_(namespace, key)`` -> 列表地址 ValueList[T]
- ``set_value`` / ``delete_value`` / ``append_list`` / ``delete_list`` 构造 Write。

Python 无 TS 的 phantom type，用 ``Generic[T]`` 表达"这个地址存 T"，
运行期只携带 namespace / key / kind；类型参数仅供静态检查。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Literal, TypeAlias, TypeVar

from .types import JsonValue

T = TypeVar("T")


# ---------------------------------------------------------------------------
# 类型化地址
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Value(Generic[T]):
    namespace: str
    key: str = ""
    kind: Literal["value"] = "value"


@dataclass(frozen=True, slots=True)
class ValueList(Generic[T]):
    namespace: str
    key: str = ""
    kind: Literal["list"] = "list"


# ---------------------------------------------------------------------------
# 读取结果 / 读取选项（对齐官方 values.ts:32-57）
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StoredValue(Generic[T]):
    address: Value[T]
    value: T
    seq: int


@dataclass(frozen=True, slots=True)
class ListElement(Generic[T]):
    seq: int
    value: T


@dataclass(frozen=True, slots=True)
class ListCursor:
    seq: int


@dataclass(frozen=True, slots=True)
class ListReadOptions:
    cursor: ListCursor | None = None
    order: Literal["asc", "desc"] = "asc"
    limit: int | None = None


@dataclass(frozen=True, slots=True)
class ResolvedListReadOptions:
    order: Literal["asc", "desc"]
    limit: int
    cursor: ListCursor | None = None


_LIST_READ_DEFAULT_LIMIT = 2**31 - 1


def resolve_list_read_options(
    options: ListReadOptions | None = None,
) -> ResolvedListReadOptions:
    if options is None:
        return ResolvedListReadOptions(order="asc", limit=_LIST_READ_DEFAULT_LIMIT)
    limit = options.limit if options.limit is not None else _LIST_READ_DEFAULT_LIMIT
    if limit < 0:
        limit = 0
    return ResolvedListReadOptions(
        order=options.order, limit=limit, cursor=options.cursor
    )


def _validate_address(namespace: str, key: str) -> None:
    if len(namespace) == 0:
        raise ValueError("Value namespace must not be empty")
    if "\u0000" in namespace:
        raise ValueError("Value namespace must not contain NUL")
    if "\u0000" in key:
        raise ValueError("Value key must not contain NUL")


def value(namespace: str, key: str = "") -> Value[T]:
    _validate_address(namespace, key)
    return Value(namespace=namespace, key=key)


def list_(namespace: str, key: str = "") -> ValueList[T]:
    _validate_address(namespace, key)
    return ValueList(namespace=namespace, key=key)


# ---------------------------------------------------------------------------
# Write 操作（供 Storage.commit 消费；实际落库在 A1/A2）
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ValueSetWrite:
    namespace: str
    key: str
    value: JsonValue
    kind: Literal["value"] = "value"
    op: Literal["set"] = "set"


@dataclass(slots=True)
class ValueDeleteWrite:
    namespace: str
    key: str
    kind: Literal["value"] = "value"
    op: Literal["delete"] = "delete"


@dataclass(slots=True)
class ListAppendWrite:
    namespace: str
    key: str
    value: JsonValue
    kind: Literal["list"] = "list"
    op: Literal["append"] = "append"


@dataclass(slots=True)
class ListDeleteWrite:
    namespace: str
    key: str
    kind: Literal["list"] = "list"
    op: Literal["delete"] = "delete"


ValueWrite: TypeAlias = ValueSetWrite | ValueDeleteWrite
ListWrite: TypeAlias = ListAppendWrite | ListDeleteWrite
# 侧存储写联合。条目级写（entry append）+ CommitResult 属 A1（Storage.commit）。
Write: TypeAlias = ValueWrite | ListWrite


def set_value(address: Value[T], next_value: T) -> ValueSetWrite:
    return ValueSetWrite(
        namespace=address.namespace, key=address.key, value=next_value  # type: ignore[arg-type]
    )


def delete_value(address: Value[T]) -> ValueDeleteWrite:
    return ValueDeleteWrite(namespace=address.namespace, key=address.key)


def append_list(address: ValueList[T], element: T) -> ListAppendWrite:
    return ListAppendWrite(
        namespace=address.namespace, key=address.key, value=element  # type: ignore[arg-type]
    )


def delete_list(address: ValueList[T]) -> ListDeleteWrite:
    return ListDeleteWrite(namespace=address.namespace, key=address.key)


# ---------------------------------------------------------------------------
# 具名地址（对齐官方 values.ts 末尾的常用地址工厂）
# ---------------------------------------------------------------------------

# LaneConfiguration = {model, thinkingLevel, activeToolNames}；配置恢复读这一处。
def lane_config(lane: str) -> Value[JsonValue]:
    return value("pi.lane.config", lane)


def branch_tip(branch: str) -> Value[str | None]:
    return value("pi.branch.tip", branch)


def session_name() -> Value[str]:
    return value("pi.session.name")


def entry_label(entry_id: str) -> Value[str]:
    return value("pi.entry.label", entry_id)


__all__ = [
    "Value",
    "ValueList",
    "StoredValue",
    "ListElement",
    "ListCursor",
    "ListReadOptions",
    "ResolvedListReadOptions",
    "resolve_list_read_options",
    "value",
    "list_",
    "ValueSetWrite",
    "ValueDeleteWrite",
    "ListAppendWrite",
    "ListDeleteWrite",
    "ValueWrite",
    "ListWrite",
    "Write",
    "set_value",
    "delete_value",
    "append_list",
    "delete_list",
    "lane_config",
    "branch_tip",
    "session_name",
    "entry_label",
]
