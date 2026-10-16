"""快速排序（Quicksort）实现。

提供两种风格：

1. ``quicksort``       —— 返回新列表的函数式写法，代码简洁易读。
2. ``quicksort_inplace``—— 原地排序（Lomuto 分区 + 三数取中 + 小数组插排），
                          空间复杂度 O(log n)，适合对已有列表高效排序。

两种实现均为平均 O(n log n)，最坏 O(n^2)（原地版本通过三数取中 +
尾递归消除降低了最坏情况出现的概率与栈深度）。

直接运行本文件会执行内置的自测：
    python quicksort.py
"""

from __future__ import annotations

import random
from typing import List, MutableSequence, Sequence, TypeVar

T = TypeVar("T")

_INSERTION_THRESHOLD = 16  # 小数组改用插入排序，减少递归开销


# --------------------------------------------------------------------------- #
# 实现一：函数式写法
# --------------------------------------------------------------------------- #
def quicksort(items: Sequence[T]) -> List[T]:
    """返回 ``items`` 升序排序后的新列表。

    以中间元素为基准，用列表推导式完成分区。可读性优先。
    """
    if len(items) <= 1:
        return list(items)

    pivot = items[len(items) // 2]
    less = [x for x in items if x < pivot]
    equal = [x for x in items if x == pivot]
    greater = [x for x in items if x > pivot]
    return quicksort(less) + equal + quicksort(greater)


# --------------------------------------------------------------------------- #
# 实现二：原地排序
# --------------------------------------------------------------------------- #
def _median_of_three(a: MutableSequence[T], lo: int, hi: int) -> T:
    """取 a[lo]、a[mid]、a[hi] 的中位数作为基准（三数取中）。"""
    mid = (lo + hi) // 2
    x, y, z = a[lo], a[mid], a[hi]
    if x < y:
        if y < z:
            return y
        return z if x < z else x
    if x < z:
        return x
    return z if y < z else y


def _insertion_sort(a: MutableSequence[T], lo: int, hi: int) -> None:
    """对闭区间 ``a[lo..hi]`` 做插入排序。"""
    for i in range(lo + 1, hi + 1):
        key = a[i]
        j = i - 1
        while j >= lo and a[j] > key:
            a[j + 1] = a[j]
            j -= 1
        a[j + 1] = key


def _partition(a: MutableSequence[T], lo: int, hi: int) -> int:
    """Lomuto 分区，返回基准的最终下标。"""
    pivot = _median_of_three(a, lo, hi)
    i = lo
    # 先把所有 < pivot 的元素换到左侧
    for j in range(lo, hi + 1):
        if a[j] < pivot:
            a[i], a[j] = a[j], a[i]
            i += 1
    # 再把 == pivot 的元素紧随其后，保证与基准相等的元素聚集在中间
    for j in range(i, hi + 1):
        if a[j] == pivot:
            a[i], a[j] = a[j], a[i]
            i += 1
    return i - 1


def quicksort_inplace(items: MutableSequence[T]) -> MutableSequence[T]:
    """原地升序排序 ``items`` 并返回它本身（会修改入参）。"""
    stack = [(0, len(items) - 1)]
    while stack:
        lo, hi = stack.pop()
        if hi - lo + 1 <= _INSERTION_THRESHOLD:
            if lo < hi:
                _insertion_sort(items, lo, hi)
            continue

        p = _partition(items, lo, hi)
        # 用显式栈模拟递归：先压大的区间，栈深度保持 O(log n)
        if p - 1 - lo > hi - (p + 1):
            if lo < p - 1:
                stack.append((lo, p - 1))
            if p + 1 < hi:
                stack.append((p + 1, hi))
        else:
            if p + 1 < hi:
                stack.append((p + 1, hi))
            if lo < p - 1:
                stack.append((lo, p - 1))
    return items


# --------------------------------------------------------------------------- #
# 自测
# --------------------------------------------------------------------------- #
def _self_test() -> None:
    cases: List[List[int]] = [
        [],
        [1],
        [2, 1],
        [3, 3, 3],
        [5, 4, 3, 2, 1],
        [1, 2, 3, 4, 5],
        [0, -1, 5, 3, 3, -7, 100, 42],
        list(range(200, 0, -1)),  # 覆盖插入排序阈值以上的路径
    ]

    for data in cases:
        expected = sorted(data)
        assert quicksort(data) == expected, f"quicksort 失败: {data}"

        buf = list(data)
        assert quicksort_inplace(buf) == expected, f"quicksort_inplace 失败: {data}"

    # 随机对拍
    rng = random.Random(20240607)
    for _ in range(200):
        data = [rng.randint(-50, 50) for _ in range(rng.randint(0, 120))]
        buf = list(data)
        assert quicksort(data) == sorted(data)
        assert quicksort_inplace(buf) == sorted(data)

    # 原地排序必须真的改的是同一个对象
    buf = [3, 1, 2]
    assert quicksort_inplace(buf) is buf and buf == [1, 2, 3]

    # 字符串 / 浮点等可比较类型
    assert quicksort(["pear", "apple", "banana"]) == ["apple", "banana", "pear"]
    assert quicksort_inplace([2.5, 1.0, 3.5]) == [1.0, 2.5, 3.5]

    print("全部自测通过 ✅")


def _demo() -> None:
    data = [10, 7, 8, 9, 1, 5, 3, 3, -2, 0]
    print("原始:", data)
    print("函数式:", quicksort(data))
    buf = list(data)
    quicksort_inplace(buf)
    print("原地   :", buf)


if __name__ == "__main__":
    _demo()
    _self_test()
