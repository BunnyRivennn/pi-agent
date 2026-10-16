"""E1 hook 注册表测试：注册/注销 + 三种聚合语义 + 三种错误策略。"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from pi_agent.agent_core.hooks import HookDeclined, HookRegistry, OnError


def _handler(result: dict[str, Any] | None):
    """返回固定结果的同步 handler，并记录它看到的事件。"""
    seen: list[dict[str, Any]] = []

    def handler(event: dict[str, Any], _abort: asyncio.Event | None):
        seen.append(event)
        return result

    handler.seen = seen  # type: ignore[attr-defined]
    return handler


def _boom(message: str = "炸了"):
    def handler(_event: dict[str, Any], _abort: asyncio.Event | None):
        raise RuntimeError(message)

    return handler


# ---------------------------------------------------------------------------
# 注册与注销
# ---------------------------------------------------------------------------


def test_unregistered_has_false() -> None:
    reg = HookRegistry()
    assert reg.has("before_compaction") is False


def test_on_returns_closure_removes_itself() -> None:
    reg = HookRegistry()
    a = _handler({"summary": "A"})
    off_a = reg.on("before_compaction", a)
    reg.on("before_compaction", _handler({"summary": "B"}))

    off_a()
    assert reg.has("before_compaction") is True  # B 还在
    off_a()  # 幂等：重复注销不抛


def test_after_close_cannot_register() -> None:
    reg = HookRegistry()
    reg.close(RuntimeError("已关闭"))
    with pytest.raises(RuntimeError, match="已关闭"):
        reg.on("after_response", _handler(None))


# ---------------------------------------------------------------------------
# fold：链式折叠
# ---------------------------------------------------------------------------


async def test_fold_next_handler_receives_previous_output() -> None:
    reg = HookRegistry()

    def first(event: dict[str, Any], _abort: asyncio.Event | None):
        return {"text": event["text"] + "-first"}

    def second(event: dict[str, Any], _abort: asyncio.Event | None):
        return {"text": event["text"] + "-second"}

    reg.on("after_response", first)
    reg.on("after_response", second)

    result = await reg.fold("after_response", {"text": "原"}, keys=["text"])
    assert result == {"text": "原-first-second"}


async def test_fold_return_none_skip_use_previous() -> None:
    reg = HookRegistry()
    reg.on("after_response", _handler({"text": "改过"}))
    reg.on("after_response", _handler(None))

    result = await reg.fold("after_response", {"text": "原"}, keys=["text"])
    assert result == {"text": "改过"}


async def test_fold_only_fold_declared_keys() -> None:
    reg = HookRegistry()
    reg.on("after_response", _handler({"text": "改过", "无关": "忽略"}))

    result = await reg.fold("after_response", {"text": "原"}, keys=["text"])
    assert result == {"text": "改过"}


async def test_fold_supports_async_handler() -> None:
    reg = HookRegistry()

    async def handler(event: dict[str, Any], _abort: asyncio.Event | None):
        await asyncio.sleep(0)
        return {"text": event["text"] + "-async"}

    reg.on("after_response", handler)
    result = await reg.fold("after_response", {"text": "原"}, keys=["text"])
    assert result == {"text": "原-async"}


# ---------------------------------------------------------------------------
# collect：累加收集
# ---------------------------------------------------------------------------


async def test_collect_accumulates_each_handler_output() -> None:
    reg = HookRegistry()
    reg.on("before_run", _handler({"messages": ["a"]}))
    reg.on("before_run", _handler({"messages": ["b", "c"]}))

    result = await reg.collect("before_run", {"messages": ["原"]}, key="messages")
    assert result == ["a", "b", "c"]  # 只返回新增的，不含原始


async def test_collect_subsequent_handler_sees_collected() -> None:
    reg = HookRegistry()
    first = _handler({"messages": ["注入"]})
    second = _handler(None)
    reg.on("before_run", first)
    reg.on("before_run", second)

    await reg.collect("before_run", {"messages": ["原"]}, key="messages")
    assert second.seen[0]["messages"] == ["原", "注入"]  # type: ignore[attr-defined]


async def test_collect_读写键名可以不同() -> None:
    """``before_run`` 就是这样：事件里读 ``prompt``，结果里写 ``messages``。"""
    reg = HookRegistry()
    handler = _handler({"messages": ["注入"]})
    reg.on("before_run", handler)

    result = await reg.collect(
        "before_run", {"prompt": ["原"]}, key="prompt", result_key="messages"
    )
    assert result == ["注入"]
    assert handler.seen[0]["prompt"] == ["原"]  # type: ignore[attr-defined]


async def test_run_before_run_读prompt写messages() -> None:
    reg = HookRegistry()
    reg.on("before_run", _handler({"messages": ["注入"]}))

    result = await reg.run_before_run({"lane": "main", "run_id": "r1", "prompt": ["原"]})
    assert result == ["注入"]


# ---------------------------------------------------------------------------
# last_wins：全跑一遍，最后表态者胜
# ---------------------------------------------------------------------------


async def test_last_wins_每个handler都被调到() -> None:
    reg = HookRegistry()
    first = _handler({"follow_up": "甲"})
    second = _handler({"follow_up": "乙"})
    reg.on("before_run_end", first)
    reg.on("before_run_end", second)

    result = await reg.last_wins("before_run_end", {}, result_key="follow_up")
    # 不短路：handler 可能有副作用（落档、提取画像），都得跑到
    assert len(first.seen) == 1  # type: ignore[attr-defined]
    assert len(second.seen) == 1  # type: ignore[attr-defined]
    assert result == "乙"


async def test_last_wins_不表态者不覆盖() -> None:
    reg = HookRegistry()
    reg.on("before_run_end", _handler({"follow_up": "甲"}))
    reg.on("before_run_end", _handler(None))

    assert await reg.last_wins("before_run_end", {}, result_key="follow_up") == "甲"


async def test_last_wins_全不表态返回None() -> None:
    reg = HookRegistry()
    reg.on("before_run_end", _handler(None))

    assert await reg.last_wins("before_run_end", {}, result_key="follow_up") is None


async def test_last_wins_某个handler炸了不影响其余() -> None:
    reg = HookRegistry()
    reg.on("before_run_end", _boom("炸了"))
    reg.on("before_run_end", _handler({"follow_up": "乙"}))

    # invokeAll 是 fail-open：一个 handler 炸了不该把整轮带走
    assert await reg.last_wins("before_run_end", {}, result_key="follow_up") == "乙"


async def test_run_before_run_end_返回followup文本() -> None:
    reg = HookRegistry()
    reg.on("before_run_end", _handler({"follow_up": "再说一点"}))

    event = {"lane": "main", "run_id": "r1", "messages": []}
    assert await reg.run_before_run_end(event) == "再说一点"  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# first_wins：首个表态者胜
# ---------------------------------------------------------------------------


async def test_first_wins_first_wins_rest_not_called() -> None:
    reg = HookRegistry()
    second = _handler({"compaction": {"summary": "不该被调用"}})
    reg.on("before_compaction", _handler({"compaction": {"summary": "赢"}}))
    reg.on("before_compaction", second)

    result = await reg.first_wins("before_compaction", {}, result_key="compaction")
    assert result is not None
    assert result["compaction"]["summary"] == "赢"
    assert second.seen == []  # type: ignore[attr-defined]  # 短路了


async def test_first_wins_abstainers_skipped() -> None:
    reg = HookRegistry()
    reg.on("before_compaction", _handler(None))
    reg.on("before_compaction", _handler({}))  # 空 dict 也是弃权
    reg.on("before_compaction", _handler({"compaction": {"summary": "最终"}}))

    result = await reg.first_wins("before_compaction", {}, result_key="compaction")
    assert result is not None
    assert result["compaction"]["summary"] == "最终"


async def test_first_wins_decline_works_and_shortcircuits() -> None:
    reg = HookRegistry()
    second = _handler({"compaction": {"summary": "不该被调用"}})
    reg.on("before_compaction", _handler({"decline": True}))
    reg.on("before_compaction", second)

    result = await reg.first_wins("before_compaction", {}, result_key="compaction")
    assert result == {"decline": True}
    assert second.seen == []  # type: ignore[attr-defined]


async def test_first_wins_contradictory_skipped_and_reported() -> None:
    """既 decline 又给 result 是矛盾的，不猜意图：跳过该 handler。"""
    reported: list[tuple[Exception, str]] = []
    reg = HookRegistry(
        report_error=lambda exc, name, _event: reported.append((exc, name))
    )
    reg.on("before_compaction", _handler({"decline": True, "compaction": {"summary": "矛盾"}}))
    reg.on("before_compaction", _handler({"compaction": {"summary": "后继"}}))

    result = await reg.first_wins("before_compaction", {}, result_key="compaction")
    assert result is not None
    assert result["compaction"]["summary"] == "后继"
    assert len(reported) == 1
    assert "cannot return both" in str(reported[0][0])


async def test_first_wins_all_abstain_returns_none() -> None:
    reg = HookRegistry()
    reg.on("before_compaction", _handler(None))
    assert await reg.first_wins("before_compaction", {}, result_key="compaction") is None


# ---------------------------------------------------------------------------
# 错误策略
# ---------------------------------------------------------------------------


async def test_fail_open_continue_on_error_use_previous() -> None:
    reported: list[str] = []
    reg = HookRegistry(report_error=lambda exc, _n, _e: reported.append(str(exc)))
    reg.on("after_response", _handler({"text": "先改"}))
    reg.on("after_response", _boom())
    reg.on("after_response", _handler({"text": "后改"}))

    result = await reg.fold("after_response", {"text": "原"}, keys=["text"])
    assert result == {"text": "后改"}  # 第三个仍跑到了
    assert reported == ["炸了"]


async def test_fail_closed_raise_to_caller() -> None:
    reg = HookRegistry()
    reg.on("transform_context", _boom("中断"))

    with pytest.raises(HookDeclined) as exc_info:
        await reg.fold(
            "transform_context", {"text": "原"}, keys=["text"], on_error=OnError.CLOSED
        )
    assert exc_info.value.hook == "transform_context"
    assert isinstance(exc_info.value.cause, RuntimeError)


async def test_fold_produced_only_无人表态时回空() -> None:
    reg = HookRegistry()
    reg.on("after_tool", _handler(None))

    produced = await reg.fold(
        "after_tool",
        {"content": "原文", "is_error": False},
        keys=("content", "is_error"),
        produced_only=True,
    )
    # 回填原值会让调用方把 is_error=False 当成 hook 的判断去用
    assert produced == {}


async def test_fold_produced_only_只回被改过的键() -> None:
    reg = HookRegistry()
    reg.on("after_tool", _handler({"content": "改过"}))

    produced = await reg.fold(
        "after_tool",
        {"content": "原文", "is_error": False},
        keys=("content", "is_error"),
        produced_only=True,
    )
    assert produced == {"content": "改过"}


async def test_fold_result_keys_只收不灌() -> None:
    reg = HookRegistry()
    second = _handler(None)
    reg.on("after_tool", _handler({"terminate": True}))
    reg.on("after_tool", second)

    produced = await reg.fold(
        "after_tool",
        {"content": "原文"},
        keys=("content",),
        result_keys=("terminate",),
        produced_only=True,
    )
    assert produced == {"terminate": True}
    # 一个 handler 要求终止，不该让后面的看见并据此改行为
    assert "terminate" not in second.seen[0]  # type: ignore[attr-defined]


async def test_run_after_tool_无人表态返回None() -> None:
    reg = HookRegistry()
    reg.on("after_tool", _handler(None))

    event = {
        "lane": "main",
        "run_id": "r1",
        "tool_call_id": "c1",
        "tool_name": "bash",
        "args": {},
        "content": "原文",
        "is_error": False,
    }
    assert await reg.run_after_tool(event) is None  # type: ignore[arg-type]


async def test_run_after_tool_只回被改过的键() -> None:
    reg = HookRegistry()
    reg.on("after_tool", _handler({"is_error": True}))

    event = {
        "lane": "main",
        "run_id": "r1",
        "tool_call_id": "c1",
        "tool_name": "bash",
        "args": {},
        "content": "原文",
        "is_error": False,
    }
    out = await reg.run_after_tool(event)  # type: ignore[arg-type]
    assert out == {"is_error": True}
    assert "content" not in out  # 没人动过的字段不回填


async def test_fail_to_block_exception_to_decline_shortcircuit() -> None:
    """工具 hook 报错不能放行——必须挡住。"""
    reg = HookRegistry()
    never = _handler({"args": {"x": 2}})
    reg.on("before_tool", _boom("权限不足"))
    reg.on("before_tool", never)

    result = await reg.fold_with_block("before_tool", {"args": {"x": 1}}, keys=["args"])
    assert result["block"] == {"reason": "权限不足"}
    assert "args" not in result  # 没人改过就不出现，调用方沿用原参数
    assert never.seen == []  # type: ignore[attr-defined]  # 短路了


async def test_fold_with_block_explicit_block_shortcircuits() -> None:
    reg = HookRegistry()
    never = _handler({"args": {"x": 3}})
    reg.on("before_tool", _handler({"args": {"x": 2}, "block": {"reason": "拒绝"}}))
    reg.on("before_tool", never)

    result = await reg.fold_with_block("before_tool", {"args": {"x": 1}}, keys=["args"])
    assert result["block"] == {"reason": "拒绝"}
    assert result["args"] == {"x": 2}  # block 前的修改仍保留
    assert never.seen == []  # type: ignore[attr-defined]


async def test_fold_with_block_no_block_when_no_intercept() -> None:
    reg = HookRegistry()
    reg.on("before_tool", _handler({"args": {"x": 2}}))

    result = await reg.fold_with_block("before_tool", {"args": {"x": 1}}, keys=["args"])
    assert "block" not in result
    assert result["args"] == {"x": 2}


# ---------------------------------------------------------------------------
# 遍历期间改注册表
# ---------------------------------------------------------------------------


async def test_handler_self_remove_does_not_affect_current_iteration() -> None:
    reg = HookRegistry()
    calls: list[str] = []

    def self_removing(_event: dict[str, Any], _abort: asyncio.Event | None):
        calls.append("first")
        off()
        return None

    off = reg.on("after_response", self_removing)

    def second(_event: dict[str, Any], _abort: asyncio.Event | None):
        calls.append("second")
        return None

    reg.on("after_response", second)

    await reg.fold("after_response", {"text": "原"}, keys=["text"])
    assert calls == ["first", "second"]  # 本轮快照不受影响
    assert reg.has("after_response") is True
