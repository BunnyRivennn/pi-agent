"""Bare-HTTP SSE adapter for the Volcengine Ark Coding Plan `/responses` endpoint.

Why this exists
---------------
Ark's coding gateway (https://ark.cn-beijing.volces.com/api/coding/v3) speaks the
Responses *protocol* (paths, auth, event names are all compatible), but its
`response.created` SSE event omits the `response.output` field (and nested
`content` fields). The openai-python SDK builds a response snapshot from that
first event and then does `snapshot.output.append(...)`, which crashes with
`'NoneType' object has no attribute 'append'`.

pi-agent itself parses raw event dicts in `_apply_openai_stream_event` and does
NOT rely on the SDK snapshot. So we bypass the SDK entirely: stream SSE over
httpx and hand plain dicts to pi-agent through the provider's `stream_request_fn`
injection point — no changes to library code needed.

Usage
-----
    from ark_responses_sse import build_ark_registry
    registry = build_ark_registry()
    Model(provider="openai", api="openai", base_url=...)  # now works over SSE
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator, Mapping

import httpx

from pi_agent.pi_ai.providers import OpenAIResponsesProvider
from pi_agent.pi_ai.registry import ProviderRegistry


async def ark_sse_stream_request(
    payload: dict,
    api_key: str,
    base_url: str | None,
) -> AsyncIterator[Mapping[str, object]]:
    """stream_request_fn replacement: raw SSE → event dicts, no SDK state machine."""
    if not base_url:
        raise ValueError("base_url is required for the Ark SSE adapter")

    url = base_url.rstrip("/") + "/responses"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }

    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=30.0)) as client:
        async with client.stream(
            "POST", url, headers=headers, json={**payload, "stream": True}
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if not data or data == "[DONE]":
                    continue
                yield json.loads(data)


def build_ark_registry() -> ProviderRegistry:
    """Registry whose `openai` Responses provider streams over raw SSE."""
    registry = ProviderRegistry()
    responses_provider = OpenAIResponsesProvider(stream_request_fn=ark_sse_stream_request)
    registry.register("openai", responses_provider)
    registry.register("openai-responses", responses_provider)
    return registry


def ark_configured() -> bool:
    return bool(os.getenv("OPENAI_API_KEY")) and "volces.com" in (
        os.getenv("OPENAI_API_BASE_URL", "")
    )
