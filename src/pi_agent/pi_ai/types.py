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

"""
使用dg-piagent skill创建一个DataAgent，要求：
1. 基于web应用，输入数据问题，Agent执行sql返回结果；
2. 这是个demo，数据你自己造，使用电商场景，2-4个表，使用sqlite；
3. 在执行sql前，要判断是否有drop、del之类的危险命令，有则进行拦截；
4. 验收方式：你自己构造几个实例验证通过，并告诉我如何验证；
5. 后端的开发语言要求：Python，前端的开发语言你自己决定；
6. 使用这个模型：openai接口

api-key：ark-xxxx
baseUrl：https://ark.cn-beijing.volces.com/api/coding/v3
model_name：ark-code-latest

7. 注意点：
    （1）后端彻底是Python语言，这种情况，你就要做翻译了（js->Python）
    （2）思路完全要和官方的js行为一致
"""