"""长期记忆 / 用户画像中间件。

这是真中间件：卸掉它 agent 照跑，只是不再记得上一个会话里的事；而摘要器卸掉
压缩就停了，那是实现。两者的分界线见本包 ``__init__`` 的说明。

两头都要挂，缺一不可：

- **注入**（读）``before_run``
  每轮开头把画像作为一条消息注入 prompt。``collect`` 语义：各 handler 注入的
  消息拼接，不互相覆盖。
- **提取**（写）``before_run_end``
  一次 run 的工具循环全跑完时触发，``event["messages"]`` 是本轮完整消息。挂这
  里而不是 ``after_response``——后者在工具循环里一轮能触发十几次，也不是
  ``before_compaction``——短会话压根不压，那条路可能永远不走。

状态存在 ``ProfileStore`` 后面，本模块不碰存储细节。默认给了个 JSON 文件实现，
换 sqlite / 向量库只要实现同一个 Protocol。

提取逻辑目前是规则版（``rule_extract``），只认几个最直白的句式。等画像字段梳理
清楚了换成模型调用，形状参考 ``pi_agent.summarizer.LlmSummarizer``——换的时候只
动 ``extract=`` 这一个参数，主干不用碰。
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from ..agent_core.hooks import (
    BeforeRunEndEvent,
    BeforeRunEndOut,
    BeforeRunEvent,
    BeforeRunOut,
)
from ..agent_core.types import UserMessage

if TYPE_CHECKING:
    import asyncio

    from ..agent_core.hooks import HookRegistry


class ProfileStore(Protocol):
    """画像存储。实现方负责持久化，中间件只管读写。"""

    def load(self, scope: str) -> str | None:
        """取画像文本；没有则 None。"""
        ...

    def save(self, scope: str, profile: str) -> None:
        """覆盖写入画像文本。"""
        ...


class JsonProfileStore:
    """最简实现：一个 JSON 文件，``{scope: profile}``。

    同步读写，调用点在 hook 里（已经在 await 链上，但这里是纯本地 IO，量极小，
    不值得为它引入 aiofiles）。
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def _read_all(self) -> dict[str, str]:
        if not self._path.exists():
            return {}
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            # 读不出来当没有，但不删原文件——坏掉的画像还能人工看。
            return {}
        return data if isinstance(data, dict) else {}

    def load(self, scope: str) -> str | None:
        value = self._read_all().get(scope)
        return value if isinstance(value, str) and value else None

    def save(self, scope: str, profile: str) -> None:
        data = self._read_all()
        data[scope] = profile
        self._path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# 从原文里提炼画像的回调：收到（本轮消息, 已有画像），返回新画像文本。
# 返回 None 表示「这批没什么可记的」，保持原画像不变。
ExtractProfile = Callable[[list[Any], str | None], Any]

# 规则版认的句式。故意只认最直白的几条——宁可漏记，不要记错：
# 记错了会被当成事实注入每一轮，比没记更难发现。
_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"我(?:叫|的名字是|名字叫)\s*([^\s，。,.!！?？]{1,20})"), "称呼"),
    (re.compile(r"我(?:喜欢|爱|偏好|习惯用)\s*([^，。,.!！?？\n]{1,40})"), "偏好"),
    (re.compile(r"我(?:不喜欢|讨厌|不想|别再)\s*([^，。,.!！?？\n]{1,40})"), "回避"),
    (re.compile(r"我(?:是|在做|的工作是)\s*([^，。,.!！?？\n]{1,40})"), "身份"),
)


def _message_text(message: Any) -> str:
    """把一条消息榨成纯文本。认不出的结构返回空串（不猜）。"""
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = [block.text for block in content if isinstance(getattr(block, "text", None), str)]
    return "\n".join(parts)


def rule_extract(messages: list[Any], existing: str | None) -> str | None:
    """规则版提取：扫用户消息里的几个固定句式，合并进已有画像。

    只扫 user 消息——assistant 说的话是模型自己的推测，记进画像等于让它自我强化。

    合并是**追加**而非覆盖：已有的行一条都不丢。画像是攒出来的，这轮没提到的
    事实不等于不成立；覆盖式写入会让「上次说过的」在这轮静默消失。
    """
    lines = [line for line in (existing or "").splitlines() if line.strip()]
    seen = set(lines)
    added = False

    for message in messages:
        if getattr(message, "role", None) != "user":
            continue
        text = _message_text(message)
        if not text:
            continue
        for pattern, label in _RULES:
            for match in pattern.finditer(text):
                value = match.group(1).strip()
                if not value:
                    continue
                line = f"- {label}：{value}"
                if line in seen:
                    continue
                seen.add(line)
                lines.append(line)
                added = True

    if not added:
        return None  # 这轮没新东西，保持原画像
    return "\n".join(lines)


class MemoryMiddleware:
    """长期记忆：提取用户画像并在每轮注入。

    用法::

        mw = MemoryMiddleware(
            store=JsonProfileStore("~/.pi-agent/profiles.json"),
            scope="xiaohong",
        )
        mw.install(sess.hooks)

    挂上就生效——两个插入点（``before_run`` / ``before_run_end``）都在
    ``agent_loop.py`` 里接好了，不挂则循环里 ``has()`` 恒假，零开销。
    """

    def __init__(
        self,
        *,
        store: ProfileStore,
        extract: ExtractProfile = rule_extract,
        scope: str = "default",
    ) -> None:
        self._store = store
        self._extract = extract
        self._scope = scope
        self._extractions = 0

    @property
    def extractions(self) -> int:
        """已完成的提取次数（画像真被改写的次数）。"""
        return self._extractions

    # -- 安装 ----------------------------------------------------------------

    def install(self, hooks: HookRegistry, *, id: str | None = None) -> Callable[[], None]:
        """注册两个 hook，返回一次性取消注册的闭包。"""
        off_run = hooks.on("before_run", self.before_run, id=id)
        off_run_end = hooks.on("before_run_end", self.before_run_end, id=id)

        def unsubscribe() -> None:
            off_run()
            off_run_end()

        return unsubscribe

    # -- 注入（读）-----------------------------------------------------------

    async def before_run(
        self,
        event: BeforeRunEvent,
        abort_event: asyncio.Event | None = None,
    ) -> BeforeRunOut | None:
        """每轮开头把画像注入 prompt。

        collect 语义：返回的消息会拼到 prompt 上，不覆盖别人注入的。没有画像就
        返回 None（不表态），不要返回空列表——语义上是两回事。

        注意读写键名不同：事件里读 ``prompt``，结果里写 ``messages``。
        """
        profile = self._store.load(self._scope)
        if not profile:
            return None
        # TODO 这条消息的角色与包装待定：现在是一条普通 user 消息，可能更该走
        # transform_context 塞进 system_prompt（画像是设定，不是用户说的话）。
        # 等 transform_context 接上后再定，别两边都塞导致重复。
        return {"messages": [UserMessage(content=f"<user-profile>\n{profile}\n</user-profile>")]}

    # -- 提取（写）-----------------------------------------------------------

    async def before_run_end(
        self,
        event: BeforeRunEndEvent,
        abort_event: asyncio.Event | None = None,
    ) -> BeforeRunEndOut | None:
        """一轮跑完，从本轮完整消息里提取画像。

        **始终返回 None**：这里只是搭车看一眼本轮消息，要不要追问不是它的事。
        返回 ``follow_up`` 会让 agent 多跑一个 turn。
        """
        messages = event["messages"]
        if messages:
            await self._extract_from(messages)
        return None

    async def _extract_from(self, messages: list[Any]) -> None:
        """跑提取回调并落盘。

        提取失败不吞：悄悄跳过的表现是「agent 莫名其妙不记事」，没有任何痕迹，
        比抛出来难查得多。调用方要容错就自己在回调里兜。
        """
        existing = self._store.load(self._scope)
        result = self._extract(messages, existing)
        profile = await result if hasattr(result, "__await__") else result
        if profile is None:
            return  # 这批没什么可记的，保持原画像
        if not isinstance(profile, str) or not profile.strip():
            raise ValueError(
                f"提取回调应返回非空字符串或 None，收到 {type(profile).__name__}；"
                f"{len(messages)} 条消息的画像未被记录。"
            )
        self._store.save(self._scope, profile.strip())
        self._extractions += 1


__all__ = [
    "ExtractProfile",
    "JsonProfileStore",
    "MemoryMiddleware",
    "ProfileStore",
    "rule_extract",
]
