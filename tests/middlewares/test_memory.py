"""记忆中间件测试：注入 / 提取 / 存储 / 规则。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from pi_agent.agent_core import AssistantMessage, TextContent, UserMessage
from pi_agent.agent_core.hooks import BeforeRunEndEvent, BeforeRunEvent, HookRegistry
from pi_agent.middlewares import JsonProfileStore, MemoryMiddleware, rule_extract


def _run_event(prompt: list[Any] | None = None) -> BeforeRunEvent:
    return {"lane": "main", "run_id": "r1", "prompt": prompt or []}  # type: ignore[return-value]


def _run_end_event(messages: list[Any]) -> BeforeRunEndEvent:
    return {"lane": "main", "run_id": "r1", "messages": messages}  # type: ignore[return-value]


def _assistant(text: str = "好") -> AssistantMessage:
    return AssistantMessage(
        content=[TextContent(text=text)],
        api="mock-api",
        provider="mock-provider",
        model="mock-model",
    )


def _middleware(tmp_path: Path, extract: Any = None, **kwargs: Any) -> MemoryMiddleware:
    return MemoryMiddleware(
        store=JsonProfileStore(tmp_path / "p.json"),
        extract=extract or (lambda _msgs, _old: "画像"),
        **kwargs,
    )


# -- 存储 --------------------------------------------------------------------


def test_存储空时读出None(tmp_path: Path) -> None:
    store = JsonProfileStore(tmp_path / "p.json")
    assert store.load("default") is None


def test_存储读写往返(tmp_path: Path) -> None:
    store = JsonProfileStore(tmp_path / "p.json")
    store.save("小红", "喜欢昆明")
    assert store.load("小红") == "喜欢昆明"
    assert store.load("别人") is None


def test_存储文件坏了当没有且不删原文件(tmp_path: Path) -> None:
    path = tmp_path / "p.json"
    path.write_text("{不是 json", encoding="utf-8")
    store = JsonProfileStore(path)
    assert store.load("default") is None
    # 坏掉的画像还能人工看——不许静默删掉
    assert path.read_text(encoding="utf-8") == "{不是 json"


# -- 注入 --------------------------------------------------------------------


@pytest.mark.asyncio
async def test_无画像时不表态(tmp_path: Path) -> None:
    mw = _middleware(tmp_path)
    # None 是"不表态"，空列表是"我注入了零条"，语义不同
    assert await mw.before_run(_run_event()) is None


@pytest.mark.asyncio
async def test_有画像则注入一条消息(tmp_path: Path) -> None:
    store = JsonProfileStore(tmp_path / "p.json")
    store.save("default", "喜欢昆明")
    mw = MemoryMiddleware(store=store)

    out = await mw.before_run(_run_event())
    assert out is not None
    (message,) = out["messages"]
    assert isinstance(message, UserMessage)
    assert "喜欢昆明" in str(message.content)


@pytest.mark.asyncio
async def test_注入走collect不覆盖已有prompt(tmp_path: Path) -> None:
    store = JsonProfileStore(tmp_path / "p.json")
    store.save("default", "喜欢昆明")
    mw = MemoryMiddleware(store=store)
    hooks = HookRegistry()
    mw.install(hooks)

    original = UserMessage(content="你好")
    collected = await hooks.run_before_run(_run_event([original]))
    # collect 只回新增的那几条，原 prompt 由调用方自己拼
    assert len(collected) == 1
    assert "喜欢昆明" in str(collected[0].content)


# -- 提取 --------------------------------------------------------------------


@pytest.mark.asyncio
async def test_一轮跑完提取并落盘(tmp_path: Path) -> None:
    seen: list[Any] = []

    def extract(messages: list[Any], _old: str | None) -> str:
        seen.append(messages)
        return "新画像"

    mw = _middleware(tmp_path, extract=extract)
    assert await mw.before_run_end(_run_end_event([_assistant()])) is None
    assert len(seen) == 1
    assert mw.extractions == 1
    assert JsonProfileStore(tmp_path / "p.json").load("default") == "新画像"


@pytest.mark.asyncio
async def test_提取时能看到已有画像(tmp_path: Path) -> None:
    store = JsonProfileStore(tmp_path / "p.json")
    store.save("default", "旧画像")
    olds: list[str | None] = []

    def extract(_msgs: list[Any], old: str | None) -> str:
        olds.append(old)
        return "新画像"

    mw = MemoryMiddleware(store=store, extract=extract)
    await mw.before_run_end(_run_end_event([_assistant()]))
    assert olds == ["旧画像"]


@pytest.mark.asyncio
async def test_不返回followup(tmp_path: Path) -> None:
    mw = _middleware(tmp_path)
    # 搭车看一眼就走，要不要追问不是它的事
    assert await mw.before_run_end(_run_end_event([_assistant()])) is None


@pytest.mark.asyncio
async def test_回调返回None则保持原画像(tmp_path: Path) -> None:
    store = JsonProfileStore(tmp_path / "p.json")
    store.save("default", "旧画像")
    mw = MemoryMiddleware(store=store, extract=lambda _m, _o: None)
    await mw.before_run_end(_run_end_event([_assistant()]))
    assert store.load("default") == "旧画像"
    assert mw.extractions == 0


@pytest.mark.asyncio
async def test_回调返回空串显式报错(tmp_path: Path) -> None:
    mw = _middleware(tmp_path, extract=lambda _m, _o: "   ")
    # 空画像被当成合法结果写进去，那段信息就真没了且事后看不出
    with pytest.raises(ValueError, match="非空字符串"):
        await mw.before_run_end(_run_end_event([_assistant()]))


@pytest.mark.asyncio
async def test_支持异步提取回调(tmp_path: Path) -> None:
    async def extract(_msgs: list[Any], _old: str | None) -> str:
        return "异步画像"

    mw = _middleware(tmp_path, extract=extract)
    await mw.before_run_end(_run_end_event([_assistant()]))
    assert JsonProfileStore(tmp_path / "p.json").load("default") == "异步画像"


@pytest.mark.asyncio
async def test_无消息可提取时不调回调(tmp_path: Path) -> None:
    calls: list[int] = []
    mw = _middleware(tmp_path, extract=lambda _m, _o: calls.append(1) or "x")
    await mw.before_run_end(_run_end_event([]))
    assert calls == []


# -- 规则提取 ----------------------------------------------------------------


def test_规则认出称呼和偏好() -> None:
    profile = rule_extract([UserMessage(content="我叫小红，我喜欢昆明")], None)
    assert profile is not None
    assert "- 称呼：小红" in profile
    assert "- 偏好：昆明" in profile


def test_规则只扫user消息() -> None:
    # assistant 说的是模型自己的推测，记进画像等于让它自我强化
    assert rule_extract([_assistant("我叫小明")], None) is None


def test_规则没命中时返回None() -> None:
    assert rule_extract([UserMessage(content="今天天气不错")], None) is None


def test_规则追加而不覆盖已有画像() -> None:
    profile = rule_extract([UserMessage(content="我喜欢昆明")], "- 称呼：小红")
    assert profile is not None
    # 这轮没提到名字不等于名字不成立——覆盖式写入会让它静默消失
    assert "- 称呼：小红" in profile
    assert "- 偏好：昆明" in profile


def test_规则同一条事实不重复记() -> None:
    assert rule_extract([UserMessage(content="我喜欢昆明")], "- 偏好：昆明") is None


def test_规则读得懂文本块结构() -> None:
    message = UserMessage(content=[TextContent(text="我不喜欢加糖")])
    profile = rule_extract([message], None)
    assert profile is not None
    assert "- 回避：加糖" in profile


def test_规则认不出的内容不猜() -> None:
    class Weird:
        role = "user"
        content = 42

    assert rule_extract([Weird()], None) is None


# -- 安装 --------------------------------------------------------------------


def test_install卸掉两个hook(tmp_path: Path) -> None:
    mw = _middleware(tmp_path)
    hooks = HookRegistry()
    off = mw.install(hooks)
    assert hooks.has("before_run")
    assert hooks.has("before_run_end")

    off()
    assert not hooks.has("before_run")
    assert not hooks.has("before_run_end")
