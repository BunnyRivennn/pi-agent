from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from ..agent_core.types import AssistantStream, LlmContext, Model, ThinkingLevel


@dataclass(slots=True, frozen=True)
class PiAIRequest:
    """
    表示“一次模型请求”的完整信息。
    它把调用模型所需的参数集中在一起
    """
    model: Model
    context: LlmContext
    reasoning: ThinkingLevel | None = None
    api_key: str | None = None
    session_id: str | None = None
    thinking_budgets: Mapping[str, int] | None = None
    max_retry_delay_ms: int | None = None


class Provider(Protocol):
    async def stream(
        self,
        request: PiAIRequest,
        abort_event: asyncio.Event | None = None,
    ) -> AssistantStream: ...


"""
if __name__ == "__main__":

    request = PiAIRequest(
    model=Model(
        id="gpt-5-mini",
        provider="openai",
        api="openai",
    ),
    context=context,
    )

    stream = await provider.stream(request)

    async for event in stream:
        print(event)
"""
