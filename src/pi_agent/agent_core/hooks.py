"""Hook 注册表与聚合运行器（E1 雏形）。

对齐官方 harness ``hooks.ts``。注册表本身是机械的（名字 → 有序 handler 列表），
真正的设计在**聚合语义**：同一 hook 挂多个 handler 时，它们的产出如何合并、
某个 handler 抛异常时整体如何收场。官方 ``HookRegistry.aggregate`` 按 hook 名
分派到不同策略，本模块把那些策略提炼成三类：

聚合策略
---------
- ``fold``   : handler 串成链，后一个收到前一个的输出。对应官方
  ``transformContext`` / ``beforePayload`` / ``afterResponse`` / ``afterTool``。
- ``collect``: 各 handler 的产出累加成一个列表。对应官方 ``beforeRun`` 的
  ``[...injected, ...result.messages]``。
- ``first_wins``: 第一个给出"非空表态"的 handler 赢，其余不再调用。对应官方
  ``firstStructural``（before_compaction / before_navigation）。

错误策略
---------
- ``OPEN``  : 报错后继续，沿用上一轮的值（官方绝大多数 hook）。
- ``CLOSED``: 向调用方抛出，中断整个操作（官方仅 before_drive）。
- ``BLOCK`` : 异常转成一个"拒绝"结果（官方 before_tool 独有——工具 hook 报错
  不能放行，只能挡住）。

与官方的有意偏离：官方用 ``gate.admit()`` 把 handler 跑在中止屏障里（Gate 属于
lane/drive 运行时机制）。本项目全程手动传 ``abort_event``，故 handler 签名直接
带 ``abort_event``，不引入 Gate。
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Literal,
    NotRequired,
    TypeAlias,
    TypedDict,
    TypeVar,
    cast,
    overload,
)

HookName: TypeAlias = Literal[
    "before_run",
    "before_run_end",
    "transform_context",
    "before_payload",
    "after_response",
    "before_tool",
    "after_tool",
    "before_compaction",
]

# -- 类型契约（对齐官方 HookMap）-------------------------------------------
#
# 官方 ``HookMap[TName]["event" | "result"]`` 靠 TS 映射类型做到"按 hook 名查
# 形状"。Python 没有映射类型，等价做法是每个 hook 一对 TypedDict，再用 ``on()``
# 的 overload 把名字和 handler 签名绑在一起。TypedDict 运行时就是 dict，所以
# 聚合运行器一行不用改——这是纯类型层的收益：键名拼错由 mypy 拦，不再静默失效。
#
# 官方 ``HookInvocation<TName> = HookMap[TName]["event"] & {lane, runId}``：
# 每个事件都带 lane / runId。本项目没有 lane/drive 运行时，但保留这两个字段，
# 扩展据此区分并发中的哪一条执行流。


class _Invocation(TypedDict):
    """所有事件的公共字段（官方 ``HookInvocation`` 的交叉部分）。"""

    lane: str
    run_id: str


class BeforeRunEvent(_Invocation):
    prompt: list[Any]


class BeforeRunOut(TypedDict):
    """注意读写键名不同：事件里读 ``prompt``，结果里写 ``messages``。

    官方就是这样（hooks.ts:140-142 读 ``event.prompt`` 写 ``result.messages``，
    checkpoint.ts:54 收 ``hook?.messages``）。handler 看到的 ``prompt`` 是"原始
    prompt + 迄今已注入的"，自己只需报告新增的那几条。
    """

    messages: NotRequired[list[Any]]


class BeforeRunEndEvent(_Invocation):
    """一次完整 run 结束：工具循环全跑完，拿到本轮全部消息。

    与 ``after_response`` 的区别是触发次数：after_response 每次 LLM 响应都触发
    （工具循环里一轮能有十几次），这里一次 run 只触发一次。要对整轮对话做事
    （画像提取、落档、统计）挂这里，不要挂 after_response。
    """

    messages: list[Any]


class BeforeRunEndOut(TypedDict):
    """``follow_up`` 会作为一条新的用户消息让 agent 继续跑。

    不想继续就别带这个键（或整个返回 None）。多个 handler 都给时后写胜——
    对齐官方 hooks.ts:98-107 的 invokeAll + 覆盖赋值。
    """

    follow_up: NotRequired[str]


class TransformContextEvent(_Invocation):
    messages: list[Any]
    system_prompt: str


class TransformContextOut(TypedDict):
    messages: NotRequired[list[Any]]
    system_prompt: NotRequired[str]


class BeforePayloadEvent(_Invocation):
    payload: Any


class BeforePayloadOut(TypedDict):
    payload: NotRequired[Any]


class AfterResponseEvent(_Invocation):
    message: Any


class AfterResponseOut(TypedDict):
    message: NotRequired[Any]


class BeforeToolEvent(_Invocation):
    tool_call_id: str
    tool_name: str
    args: dict[str, Any]


class ToolBlock(TypedDict):
    reason: str
    terminate: NotRequired[bool]


class BeforeToolOut(TypedDict):
    args: NotRequired[dict[str, Any]]
    block: NotRequired[ToolBlock]


class AfterToolEvent(_Invocation):
    tool_call_id: str
    tool_name: str
    args: dict[str, Any]
    content: Any
    is_error: bool
    details: NotRequired[Any]


class AfterToolOut(TypedDict):
    content: NotRequired[Any]
    details: NotRequired[Any]
    is_error: NotRequired[bool]
    terminate: NotRequired[bool]


CompactionReason: TypeAlias = Literal["manual", "threshold", "overflow"]


class CompactionPreparation(TypedDict):
    """官方 ``CompactionPreparation`` 的本项目子集。

    官方还有 ``turn_prefix_messages`` / ``is_split_turn`` / ``file_ops``——那些
    依赖本项目尚未实现的切turn与文件操作提取，等实现了再补字段，不先占位。
    """

    messages_to_summarize: list[Any]
    retained_tail: list[Any]
    tokens_before: int
    transcript: str
    branch: str
    context_window: int


class CompactResult(TypedDict):
    summary: str
    details: NotRequired[Any]


class BeforeCompactionEvent(_Invocation):
    reason: CompactionReason
    preparation: CompactionPreparation
    custom_instructions: NotRequired[str]


class BeforeCompactionOut(TypedDict):
    decline: NotRequired[bool]
    compaction: NotRequired[CompactResult]


# handler 收到一个事件 dict 和中止信号，返回一个结果 dict（或 None 表示弃权）。
HookResult: TypeAlias = dict[str, Any] | None
HookHandler: TypeAlias = Callable[
    [Any, asyncio.Event | None],
    HookResult | Awaitable[HookResult],
]

# 错误上报：拿到异常、hook 名、事件。注册表自己不打日志，交给宿主。
HookErrorReporter: TypeAlias = Callable[[Exception, HookName, dict[str, Any]], None]


class OnError(Enum):
    """handler 抛异常时的收场方式。"""

    OPEN = "open"  # 吞掉、上报、继续
    CLOSED = "closed"  # 向调用方抛出
    BLOCK = "block"  # 转成拒绝结果


class HookDeclined(Exception):
    """``OnError.CLOSED`` 策略下 handler 抛异常时的包装异常。"""

    def __init__(self, hook: HookName, cause: Exception) -> None:
        super().__init__(f"hook {hook!r} failed: {cause}")
        self.hook = hook
        self.cause = cause


T = TypeVar("T")


async def _maybe_await(value: T | Awaitable[T]) -> T:
    if inspect.isawaitable(value):
        return await cast("Awaitable[T]", value)
    return value


@dataclass(slots=True)
class _Registration:
    handler: HookHandler
    id: str | None = None


@dataclass(slots=True)
class HookRegistry:
    """有序 hook 注册表 + 三种聚合运行器。

    ``on()`` 返回一个取消注册的闭包（对齐官方 ``on`` 的返回值），便于扩展在
    卸载时精确摘掉自己挂的那一个，而不是按名字清空。
    """

    report_error: HookErrorReporter | None = None
    _registrations: dict[HookName, list[_Registration]] = field(default_factory=dict)
    _closed: Exception | None = None

    # -- 注册 ---------------------------------------------------------------

    @overload
    def on(
        self,
        name: Literal["before_run"],
        handler: Callable[
            [BeforeRunEvent, asyncio.Event | None],
            BeforeRunOut | None | Awaitable[BeforeRunOut | None],
        ],
        *,
        id: str | None = None,
    ) -> Callable[[], None]: ...

    @overload
    def on(
        self,
        name: Literal["before_run_end"],
        handler: Callable[
            [BeforeRunEndEvent, asyncio.Event | None],
            BeforeRunEndOut | None | Awaitable[BeforeRunEndOut | None],
        ],
        *,
        id: str | None = None,
    ) -> Callable[[], None]: ...

    @overload
    def on(
        self,
        name: Literal["transform_context"],
        handler: Callable[
            [TransformContextEvent, asyncio.Event | None],
            TransformContextOut | None | Awaitable[TransformContextOut | None],
        ],
        *,
        id: str | None = None,
    ) -> Callable[[], None]: ...

    @overload
    def on(
        self,
        name: Literal["before_payload"],
        handler: Callable[
            [BeforePayloadEvent, asyncio.Event | None],
            BeforePayloadOut | None | Awaitable[BeforePayloadOut | None],
        ],
        *,
        id: str | None = None,
    ) -> Callable[[], None]: ...

    @overload
    def on(
        self,
        name: Literal["after_response"],
        handler: Callable[
            [AfterResponseEvent, asyncio.Event | None],
            AfterResponseOut | None | Awaitable[AfterResponseOut | None],
        ],
        *,
        id: str | None = None,
    ) -> Callable[[], None]: ...

    @overload
    def on(
        self,
        name: Literal["before_tool"],
        handler: Callable[
            [BeforeToolEvent, asyncio.Event | None],
            BeforeToolOut | None | Awaitable[BeforeToolOut | None],
        ],
        *,
        id: str | None = None,
    ) -> Callable[[], None]: ...

    @overload
    def on(
        self,
        name: Literal["after_tool"],
        handler: Callable[
            [AfterToolEvent, asyncio.Event | None],
            AfterToolOut | None | Awaitable[AfterToolOut | None],
        ],
        *,
        id: str | None = None,
    ) -> Callable[[], None]: ...

    @overload
    def on(
        self,
        name: Literal["before_compaction"],
        handler: Callable[
            [BeforeCompactionEvent, asyncio.Event | None],
            BeforeCompactionOut | None | Awaitable[BeforeCompactionOut | None],
        ],
        *,
        id: str | None = None,
    ) -> Callable[[], None]: ...

    def on(self, name: HookName, handler: Any, *, id: str | None = None) -> Callable[[], None]:
        """挂一个 handler，返回取消注册的闭包。按注册顺序调用。

        七个 overload 对应官方 ``on<TName extends HookName>``：名字决定 handler
        收到什么事件、能返回什么结果。写错键名、返回错形状，mypy 当场报。
        """
        if self._closed is not None:
            raise self._closed
        registrations = self._registrations.setdefault(name, [])
        registration = _Registration(handler=handler, id=id)
        registrations.append(registration)

        def unregister() -> None:
            try:
                registrations.remove(registration)
            except ValueError:
                pass  # 已被移除：幂等

        return unregister

    def has(self, name: HookName) -> bool:
        return bool(self._registrations.get(name))

    def close(self, error: Exception) -> None:
        """关闭注册表：此后 ``on()`` 抛出。已注册的 handler 仍可运行。"""
        if self._closed is None:
            self._closed = error

    def _handlers(self, name: HookName) -> list[_Registration]:
        # 复制一份：handler 内部可能注册/注销，不能边遍历边改。
        return list(self._registrations.get(name, []))

    def _report(self, error: Exception, name: HookName, event: dict[str, Any]) -> None:
        if self.report_error is not None:
            self.report_error(error, name, event)

    @staticmethod
    def _aborted(abort_event: asyncio.Event | None) -> bool:
        """中止信号已置位：停止调用后续 handler。

        官方靠 ``gate.admit()`` 在屏障里抛（Gate 属 lane/drive 运行时），本项目
        手动传 ``abort_event``，所以每个运行器在每次调用前自己查一次——否则十个
        handler 串在链上，信号半路来了，剩下九个照跑。
        """
        return abort_event is not None and abort_event.is_set()

    # -- 聚合运行器 ---------------------------------------------------------

    async def fold(
        self,
        name: HookName,
        event: dict[str, Any],
        *,
        keys: Iterable[str],
        result_keys: Iterable[str] = (),
        produced_only: bool = False,
        abort_event: asyncio.Event | None = None,
        on_error: OnError = OnError.OPEN,
    ) -> dict[str, Any]:
        """链式折叠：handler 返回的 ``keys`` 字段覆写进事件，传给下一个。

        - ``keys``：**链式**字段。会回灌进下一个 handler 的事件，所以后面的人
          看到的是前面改过的值。
        - ``result_keys``：只收不灌。``after_tool`` 的 ``terminate`` 是这种——
          一个 handler 要求终止，不该让后面的 handler 看见并据此改行为。
        - ``produced_only``：只返回真被 handler 写过的键（官方 ``afterTool`` 的
          ``aggregate``）。默认 False 返回全部跟踪字段并回填原值（官方
          ``transformContext`` / ``beforePayload`` / ``afterResponse`` 就是这样）。

        handler 返回 None 或不含某个 key，视为该字段弃权、沿用上一轮的值。

        ``produced_only`` 不是风格选择：工具结果那几个字段里，「没人改」和
        「有人改成了跟原值一样」必须区分开。回填原值会让调用方以为 hook 表过
        态，把 ``is_error`` 这类字段当成 hook 的判断去用，于是工具到底成没成
        功就被误判了。
        """
        tracked = list(keys)
        collected = list(result_keys)
        current = {key: event.get(key) for key in tracked}
        produced: dict[str, Any] = {}
        for registration in self._handlers(name):
            if self._aborted(abort_event):
                break
            payload = {**event, **current}
            try:
                result = await _maybe_await(registration.handler(payload, abort_event))
            except Exception as exc:  # noqa: BLE001
                self._report(exc, name, payload)
                if on_error is OnError.CLOSED:
                    raise HookDeclined(name, exc) from exc
                continue
            if not result:
                continue
            for key in tracked:
                if key in result:
                    current[key] = result[key]
                    produced[key] = result[key]
            for key in collected:
                if key in result:
                    produced[key] = result[key]
        return produced if produced_only else current

    async def collect(
        self,
        name: HookName,
        event: dict[str, Any],
        *,
        key: str,
        result_key: str | None = None,
        abort_event: asyncio.Event | None = None,
        on_error: OnError = OnError.OPEN,
    ) -> list[Any]:
        """累加收集：各 handler 返回的列表拼接。

        ``key`` 是事件里被累加的字段，``result_key`` 是 handler 返回结果里的
        键（默认同 ``key``）。两者可以不同——官方 before_run 事件里读
        ``prompt``、结果里写 ``messages``（hooks.ts:140-142）。

        每个 handler 看到的事件里，``key`` 是"原始值 + 迄今已收集的"，所以它
        只需报告自己新增的那几条，不必关心别人注入了什么。
        """
        out_key = result_key if result_key is not None else key
        base = list(event.get(key) or [])
        collected: list[Any] = []
        for registration in self._handlers(name):
            if self._aborted(abort_event):
                break
            payload = {**event, key: [*base, *collected]}
            try:
                result = await _maybe_await(registration.handler(payload, abort_event))
            except Exception as exc:  # noqa: BLE001
                self._report(exc, name, payload)
                if on_error is OnError.CLOSED:
                    raise HookDeclined(name, exc) from exc
                continue
            if not result:
                continue
            produced = result.get(out_key)
            if produced:
                collected.extend(produced)
        return collected

    async def last_wins(
        self,
        name: HookName,
        event: dict[str, Any],
        *,
        result_key: str,
        abort_event: asyncio.Event | None = None,
        on_error: OnError = OnError.OPEN,
    ) -> Any | None:
        """全跑一遍，最后一个表态的赢。

        和 ``first_wins`` 的区别：这里不短路，每个 handler 都会被调到（它们可
        能有副作用——落档、提取画像），只是 ``result_key`` 的值被后来者覆盖。
        对齐官方 ``invokeAll`` + 覆盖赋值（hooks.ts:98-107 的 before_run_end）。

        返回 None 表示全体弃权。
        """
        winner: Any | None = None
        for registration in self._handlers(name):
            if self._aborted(abort_event):
                break
            try:
                result = await _maybe_await(registration.handler(event, abort_event))
            except Exception as exc:  # noqa: BLE001
                self._report(exc, name, event)
                if on_error is OnError.CLOSED:
                    raise HookDeclined(name, exc) from exc
                continue
            if not isinstance(result, dict):
                continue
            produced = result.get(result_key)
            if produced is not None:
                winner = produced
        return winner

    async def first_wins(
        self,
        name: HookName,
        event: dict[str, Any],
        *,
        result_key: str,
        abort_event: asyncio.Event | None = None,
        on_error: OnError = OnError.OPEN,
    ) -> dict[str, Any] | None:
        """首个表态者胜：第一个返回 ``decline=True`` 或 ``result_key`` 的赢。

        两者同时给出是矛盾表态（既弃权又给结果）：上报并跳过该 handler，不猜
        意图。对齐官方 ``firstStructural``。返回 None 表示全体弃权。
        """
        for registration in self._handlers(name):
            if self._aborted(abort_event):
                break
            try:
                result = await _maybe_await(registration.handler(event, abort_event))
            except Exception as exc:  # noqa: BLE001
                self._report(exc, name, event)
                if on_error is OnError.CLOSED:
                    raise HookDeclined(name, exc) from exc
                continue
            if not isinstance(result, dict):
                continue
            declined = result.get("decline") is True
            has_result = result.get(result_key) is not None
            if declined and has_result:
                self._report(
                    ValueError(f"{name} hook cannot return both decline and {result_key}"),
                    name,
                    event,
                )
                continue
            if declined or has_result:
                return result
        return None

    async def fold_with_block(
        self,
        name: HookName,
        event: dict[str, Any],
        *,
        keys: Iterable[str],
        abort_event: asyncio.Event | None = None,
    ) -> dict[str, Any]:
        """``OnError.BLOCK`` 专用：fold + 短路拒绝（官方 beforeTool 语义）。

        任一 handler 返回 ``block`` 或抛异常，立即停止后续 handler，结果里带
        ``block={"reason": ...}``。工具 hook 报错不能当没发生——那等于放行一个
        本该被拦住的调用。
        """
        tracked = list(keys)
        current = {key: event.get(key) for key in tracked}
        produced: dict[str, Any] = {}
        block: dict[str, Any] | None = None
        for registration in self._handlers(name):
            # 中止时不合成 block：工具调用本来就不会发生，运行时自己收场。
            if self._aborted(abort_event):
                break
            payload = {**event, **current}
            try:
                result = await _maybe_await(registration.handler(payload, abort_event))
            except Exception as exc:  # noqa: BLE001
                self._report(exc, name, payload)
                block = {"reason": str(exc)}
                break
            if not result:
                continue
            for key in tracked:
                if key in result:
                    current[key] = result[key]
                    produced[key] = result[key]
            if result.get("block") is not None:
                block = result["block"]
                break
        # 只回被改过的键 + block（官方 hooks.ts:182-185：args 没变就不回）。
        # 回填原 args 会让调用方以为 hook 重写了参数，分不清「没人管」和
        # 「有人看过、决定原样放行」。
        return {**produced, **({} if block is None else {"block": block})}

    # -- 按 hook 名的具体运行器 ---------------------------------------------
    #
    # 官方注册表对外只有 ``on()``，聚合由 ``aggregate(name, event)`` 按名内部
    # 分派——调用方永远不传键名。下面七个方法就是那层分派：插入点调
    # ``run_before_tool(...)``，不再手搓 ``fold(..., keys=["args"])``。
    # 键名集中在这里一处，拼错由 mypy 拦。

    async def run_before_run(
        self,
        event: BeforeRunEvent,
        *,
        abort_event: asyncio.Event | None = None,
    ) -> list[Any]:
        """collect：返回各 handler 注入的消息（官方 beforeRun）。

        读事件的 ``prompt``、收结果的 ``messages``——官方的读写键名不同。
        """
        return await self.collect(
            "before_run",
            dict(event),
            key="prompt",
            result_key="messages",
            abort_event=abort_event,
        )

    async def run_before_run_end(
        self,
        event: BeforeRunEndEvent,
        *,
        abort_event: asyncio.Event | None = None,
    ) -> str | None:
        """last-wins：返回要追加的 follow-up 文本，没有则 None。

        所有 handler 都会被调到（它们可能有副作用），只有 ``follow_up`` 的值
        被后来者覆盖。
        """
        follow_up = await self.last_wins(
            "before_run_end",
            dict(event),
            result_key="follow_up",
            abort_event=abort_event,
        )
        return cast("str | None", follow_up)

    async def run_transform_context(
        self,
        event: TransformContextEvent,
        *,
        abort_event: asyncio.Event | None = None,
    ) -> TransformContextOut:
        """fold：链式改写 messages / system_prompt。"""
        folded = await self.fold(
            "transform_context",
            dict(event),
            keys=("messages", "system_prompt"),
            abort_event=abort_event,
        )
        return cast("TransformContextOut", folded)

    async def run_before_payload(
        self,
        event: BeforePayloadEvent,
        *,
        abort_event: asyncio.Event | None = None,
    ) -> Any:
        """fold：返回最终 payload。"""
        folded = await self.fold(
            "before_payload", dict(event), keys=("payload",), abort_event=abort_event
        )
        return folded["payload"]

    async def run_after_response(
        self,
        event: AfterResponseEvent,
        *,
        abort_event: asyncio.Event | None = None,
    ) -> Any:
        """fold：返回最终 message。"""
        folded = await self.fold(
            "after_response", dict(event), keys=("message",), abort_event=abort_event
        )
        return folded["message"]

    async def run_before_tool(
        self,
        event: BeforeToolEvent,
        *,
        abort_event: asyncio.Event | None = None,
    ) -> BeforeToolOut:
        """fold + 短路拒绝：handler 报错转成 block，不放行。

        结果里只有被改过的键：``args`` 没人重写就不出现，调用方沿用原参数。
        """
        folded = await self.fold_with_block(
            "before_tool", dict(event), keys=("args",), abort_event=abort_event
        )
        return cast("BeforeToolOut", folded)

    async def run_after_tool(
        self,
        event: AfterToolEvent,
        *,
        abort_event: asyncio.Event | None = None,
    ) -> AfterToolOut | None:
        """fold（只回被改过的键）：链式改写工具结果。

        没有 handler 表态就返回 None，不是一个回填了原值的字典——调用方据此判
        断「要不要覆盖工具的原始结果」。对齐官方 ``afterTool`` 的 ``aggregate``
        （hooks.ts:304、334：空对象时回 undefined）。

        ``terminate`` 只收不灌：某个 handler 要求终止，不该让后面的 handler 看
        见这个决定并据此改自己的行为。
        """
        produced = await self.fold(
            "after_tool",
            dict(event),
            keys=("content", "details", "is_error"),
            result_keys=("terminate",),
            produced_only=True,
            abort_event=abort_event,
        )
        return cast("AfterToolOut", produced) if produced else None

    async def run_before_compaction(
        self,
        event: BeforeCompactionEvent,
        *,
        abort_event: asyncio.Event | None = None,
    ) -> BeforeCompactionOut | None:
        """first-wins：首个 decline 或给出 compaction 的 handler 赢。"""
        decision = await self.first_wins(
            "before_compaction",
            dict(event),
            result_key="compaction",
            abort_event=abort_event,
        )
        return cast("BeforeCompactionOut | None", decision)


__all__ = [
    "AfterResponseEvent",
    "AfterResponseOut",
    "AfterToolEvent",
    "AfterToolOut",
    "BeforeCompactionEvent",
    "BeforeCompactionOut",
    "BeforePayloadEvent",
    "BeforePayloadOut",
    "BeforeRunEndEvent",
    "BeforeRunEndOut",
    "BeforeRunEvent",
    "BeforeRunOut",
    "BeforeToolEvent",
    "BeforeToolOut",
    "CompactResult",
    "CompactionPreparation",
    "CompactionReason",
    "ToolBlock",
    "TransformContextEvent",
    "TransformContextOut",
    "HookName",
    "HookHandler",
    "HookResult",
    "HookErrorReporter",
    "HookRegistry",
    "HookDeclined",
    "OnError",
]
