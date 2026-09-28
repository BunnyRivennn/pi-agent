"""同一文件的写操作串行化。

对齐官方 ``file-mutation-queue.ts``：按「解析后的真实路径」分桶加锁，**不同文件
仍然并行**。没有这层，模型并行发两个 ``edit`` 改同一文件时，后写的会整体覆盖
前一个的结果（两者都基于同一份旧内容做的替换）。

分桶键用 ``os.path.realpath``：符号链接指向同一文件时必须排到同一个队列里，
否则锁了个假名字，等于没锁。
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TypeVar

__all__ = ["with_file_mutation_lock"]

T = TypeVar("T")

# 键是 realpath 字符串，值是（锁, 引用计数）。引用计数归零时才能安全回收，
# 光看 lock.locked() 不够：正在 await 等锁的协程还没持有锁，但不能把它们的锁换掉。
_locks: dict[str, tuple[asyncio.Lock, int]] = {}
# 保护 _locks 自身的读改写。
_registry_lock = asyncio.Lock()


def _queue_key(path: Path) -> str:
    """算出分桶键；路径不存在时退回规范化的绝对路径。"""
    absolute = os.path.abspath(path)
    try:
        return os.path.realpath(absolute)
    except OSError:
        # 路径缺失/父目录不是目录时 realpath 可能失败，这在“即将创建”的写入场景很常见。
        return absolute


async def with_file_mutation_lock(path: Path, fn: Callable[[], Awaitable[T]]) -> T:
    """持有 ``path`` 的写锁执行 ``fn``。

    锁在 ``fn`` 完成（含抛异常）后释放。没人再用的锁会被回收，避免长跑进程里
    ``_locks`` 无限增长。
    """
    key = _queue_key(path)

    # 取锁（或新建）并登记一次引用，表明“我要用这把锁”。
    async with _registry_lock:
        lock, refs = _locks.get(key, (asyncio.Lock(), 0))
        _locks[key] = (lock, refs + 1)

    try:
        async with lock:
            return await fn()
    finally:
        # 退掉引用；最后一个离开的人负责清掉条目。
        async with _registry_lock:
            current = _locks.get(key)
            if current is not None and current[0] is lock:
                remaining = current[1] - 1
                if remaining <= 0:
                    _locks.pop(key, None)
                else:
                    _locks[key] = (lock, remaining)
