"""OpenRouterProvider — uses the OpenAI SDK pointed at OpenRouter's endpoint."""

from collections.abc import AsyncIterator
from decimal import Decimal

from openai import AsyncOpenAI

from agentkit._content import TextBlock
from agentkit._messages import Message, Usage
from agentkit.providers.base import (
    Provider,
    ProviderCapabilities,
    ProviderEvent,
    ProviderRequest,
)
from agentkit.providers.openrouter.pricing import estimate_cost_usd
from agentkit.providers.openrouter.request_builder import build_openrouter_request
from agentkit.providers.openrouter.stream_parser import parse_openrouter_stream


class OpenRouterProvider(Provider):
    name = "openrouter"
    capabilities = ProviderCapabilities(
        supports_tool_use=True,
        supports_parallel_tools=True,
        supports_prompt_caching=True,
        supports_vision=True,
        supports_thinking=False,  # not exposed via OpenRouter currently
        max_context_tokens=128_000,  # conservative default; varies per model
        max_output_tokens=8_192,
    )

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str = "https://openrouter.ai/api/v1",
        client: AsyncOpenAI | None = None,
        http_referer: str | None = None,
        x_title: str | None = None,
    ) -> None:
        self._client = client or AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._extra_headers: dict[str, str] = {}
        if http_referer:
            self._extra_headers["HTTP-Referer"] = http_referer
        if x_title:
            self._extra_headers["X-Title"] = x_title

    async def stream(self, request: ProviderRequest) -> AsyncIterator[ProviderEvent]:
        payload = build_openrouter_request(request)
        # Always include usage info per chunk via stream_options when supported.
        payload.setdefault("stream_options", {"include_usage": True})
        chunks = await self._client.chat.completions.create(  # type: ignore[reportUnknownVariableType]
            extra_headers=self._extra_headers or None,
            **payload,
        )
        async for ev in parse_openrouter_stream(chunks):  # type: ignore[reportUnknownArgumentType]
            yield ev

    def estimate_tokens(self, messages: list[Message]) -> int:
        # Using isinstance(b, TextBlock) instead of hasattr(b, "text") to avoid
        # accidentally counting ThinkingBlock.text (which also has a .text field).
        total = 0
        for m in messages:
            for b in m.content:
                if isinstance(b, TextBlock):
                    total += len(b.text) // 4
        return total

    def estimate_cost(self, usage: Usage) -> Decimal:
        return Decimal("0")

    def estimate_cost_for(self, model: str, usage: Usage) -> Decimal:
        return estimate_cost_usd(model, usage)
