from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Generic, TypeVar

from .types import AssistantMessage, AssistantMessageEvent

TEvent = TypeVar("TEvent")
TResult = TypeVar("TResult")


class EventStream(Generic[TEvent, TResult]):
    """Async iterable event stream with terminal result extraction."""

    def __init__(
        self,
        is_complete: Callable[[TEvent], bool],
        extract_result: Callable[[TEvent], TResult],
    ) -> None:
        self._is_complete = is_complete
        self._extract_result = extract_result
        # ① 事件管道：生产者 push，消费者 get
        self._queue: asyncio.Queue[object] = asyncio.Queue()
        # ② 终态标记：一旦完成，拒收新事件
        self._done = False
        # ③ 哨兵：一个独一无二的对象，当"流结束了"的信号
        self._sentinel = object()
        # ④ 信箱：终态结果放这里，await 它就能拿到
        self._result_future: asyncio.Future[TResult] = asyncio.get_event_loop().create_future()

    def push(self, event: TEvent) -> None:
        if self._done:
            return

        if self._is_complete(event):
            self._done = True
            if not self._result_future.done():
                self._result_future.set_result(self._extract_result(event))

        self._queue.put_nowait(event)

        if self._done:
            self._queue.put_nowait(self._sentinel)

    def end(self, result: TResult | None = None) -> None:
        if self._done:
            return

        self._done = True

        if not self._result_future.done():
            if result is None:
                self._result_future.set_exception(
                    RuntimeError("Stream ended before a terminal result was emitted")
                )
            else:
                self._result_future.set_result(result)

        self._queue.put_nowait(self._sentinel)

    async def result(self) -> TResult:
        return await self._result_future

    def __aiter__(self) -> EventStream[TEvent, TResult]:
        return self

    async def __anext__(self) -> TEvent:
        item = await self._queue.get()  # 拿到了哨兵（sentinel）
        if item is self._sentinel:
            raise StopAsyncIteration
        return item  # type: ignore[return-value]


class AssistantMessageEventStream(EventStream[AssistantMessageEvent, AssistantMessage]):
    """EventStream specialization for assistant streaming events."""

    def __init__(self) -> None:
        """
        遇到 done 或 error 类型的事件，就认定流到头了
        """
        super().__init__(
            is_complete=lambda event: event["type"] in {"done", "error"},
            extract_result=_extract_assistant_result,
        )


def _extract_assistant_result(event: AssistantMessageEvent) -> AssistantMessage:
    if event["type"] == "done":
        return event["message"]
    if event["type"] == "error":
        return event["error"]
    raise RuntimeError(f"Unexpected terminal event type: {event['type']}")
