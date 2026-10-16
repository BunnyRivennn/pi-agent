"""压缩摘要器：``session.factory.Summarize`` 的默认实现。

这不是中间件。摘要是压缩**实现**的一部分，走依赖注入——对齐官方：摘要在压缩
运行时里直接生成（compaction.ts 的 request 调用），``beforeCompaction`` 只是
干预点，不是实现入口。拿 hook 当实现入口会和已有的注入缝重复，卸掉 hook 行为
不变，那它就不是中间件。

住在顶层而不是 agent_core：摘要要调 LLM，而依赖方向是 pi_ai → agent_core，
agent_core 是叶子，不能反过来依赖 pi_ai。本模块同时用到两边（agent_core 的
提示词与消息类型 + pi_ai 的 complete_simple），所以压在它们之上。

摘要用独立的一次 LLM 调用：不带工具定义、不进主历史、调完即弃。模型可以和主
模型不同（小参数模型足够），构造时传进来。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from .agent_core.compaction import SUMMARIZATION_PROMPT, SUMMARIZATION_SYSTEM_PROMPT
from .agent_core.types import TextContent
from .pi_ai.runtime import complete_simple

if TYPE_CHECKING:
    import asyncio

    from .agent_core.types import Model
    from .pi_ai.registry import ProviderRegistry


class SummarizationError(RuntimeError):
    """摘要生成失败且无法降级。

    显式抛出而不是回一段空摘要：空摘要会被当成合法结果提交进 CompactionEntry，
    那一段上下文就真的没了，且事后看不出是哪里丢的。
    """


class LlmSummarizer:
    """起一次性 LLM 调用把会话文本压成摘要。

    直接当 ``Summarize`` 回调用::

        summarizer = LlmSummarizer(model=small_model, registry=registry)
        await create_agent_session(..., summarize=summarizer)
    """

    def __init__(
        self,
        *,
        model: Model,
        registry: ProviderRegistry,
        custom_instructions: str | None = None,
        on_progress: Callable[[str], None] | None = None,
    ) -> None:
        self._model = model
        self._registry = registry
        self._custom_instructions = custom_instructions
        self._on_progress = on_progress
        self._summaries = 0

    @property
    def summaries(self) -> int:
        """已生成的摘要条数。"""
        return self._summaries

    async def __call__(
        self,
        transcript: str,
        *,
        custom_instructions: str | None = None,
        abort_event: asyncio.Event | None = None,
    ) -> str:
        """把 transcript 压成摘要。

        ``custom_instructions`` 覆盖构造时传的那份——留给调用方按轮次改要求
        （比如中间件在 ``before_compaction`` 里决定这一轮重点保留什么）。
        """
        if not transcript.strip():
            # 空输入不去问模型：它只会编一段无来源的摘要，比报错更糟。
            raise SummarizationError("待摘要文本为空。")

        prompt = f"{transcript}\n\n{SUMMARIZATION_PROMPT}"
        instructions = custom_instructions or self._custom_instructions
        if instructions:
            prompt = f"{prompt}\n\n额外要求：\n{instructions}"

        self._report("压缩中")
        kwargs: dict[str, Any] = {}
        if abort_event is not None:
            kwargs["abort_event"] = abort_event
        message = await complete_simple(
            prompt,
            model=self._model,
            registry=self._registry,
            system_prompt=SUMMARIZATION_SYSTEM_PROMPT,
            **kwargs,
        )
        text = "\n".join(
            block.text for block in message.content if isinstance(block, TextContent)
        ).strip()
        if not text:
            # 不回空串：见 SummarizationError 的说明。
            raise SummarizationError(
                f"摘要模型返回空文本（stop_reason={message.stop_reason}）；"
                f"待摘要 {len(transcript)} 字符未被压缩。"
            )
        self._summaries += 1
        self._report("压缩完成")
        return text

    def _report(self, stage: str) -> None:
        if self._on_progress is None:
            return
        try:
            self._on_progress(stage)
        except Exception:  # noqa: BLE001, S110
            pass  # 进度上报炸了不能带走压缩


__all__ = ["LlmSummarizer", "SummarizationError"]
