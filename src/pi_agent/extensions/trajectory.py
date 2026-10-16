"""轨迹落盘：把事件流原样写成 JSONL。

挂在 ``agent.subscribe()`` 上，不经过 hooks——轨迹要的是全量（含流式 delta、
turn 边界），而 hook 只在七个点被叫到，看不到这些。而且日志不该影响数据流。

一行一个事件，带本地时间戳。事件里含 message / ToolCall 等对象，故序列化时
``default=str`` 兜底。
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from types import TracebackType
from typing import Any, Self


class TrajectoryLogger:
    """把 agent 事件逐条写成 JSONL。

    listener 是同步调用的（在 agent 的事件推送路径上），所以这里只做同步写 +
    flush，不起 task、不 await。需要异步落盘就把事件塞进队列，另起消费者。

    ``on_error`` 决定写盘失败怎么收场。默认是吞掉并计数：日志坏了不该把 agent
    带走——这跟 hook 的 fail-open 是同一个道理，只是更彻底（连上报都可选）。
    """

    def __init__(
        self,
        path: str | Path,
        *,
        on_error: Callable[[Exception, dict[str, Any]], None] | None = None,
    ) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = self._path.open("a", encoding="utf-8")
        self._on_error = on_error
        self._written = 0
        self._dropped = 0

    @property
    def path(self) -> Path:
        return self._path

    @property
    def written(self) -> int:
        """成功写入的事件条数。"""
        return self._written

    @property
    def dropped(self) -> int:
        """写盘失败而丢弃的条数。非 0 说明轨迹不完整。"""
        return self._dropped

    def __call__(self, event: dict[str, Any]) -> None:
        """subscribe 的 listener 入口。"""
        if self._fp.closed:
            self._drop(ValueError(f"轨迹文件已关闭：{self._path}"), event)
            return
        try:
            line = json.dumps(
                {"ts": time.time(), **event},
                ensure_ascii=False,
                default=str,
            )
            self._fp.write(line + "\n")
            self._fp.flush()
            self._written += 1
        except Exception as exc:  # noqa: BLE001
            self._drop(exc, event)

    def _drop(self, exc: Exception, event: dict[str, Any]) -> None:
        """记一条丢弃并上报。两条丢弃路径（已关闭 / 写失败）都走这里。"""
        self._dropped += 1
        if self._on_error is None:
            return
        try:
            self._on_error(exc, event)
        except Exception:  # noqa: BLE001, S110
            pass  # 上报自己炸了也不能往外抛——listener 抛异常会炸事件流

    def close(self) -> None:
        if not self._fp.closed:
            self._fp.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


__all__ = ["TrajectoryLogger"]
