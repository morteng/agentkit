"""AnthropicProvider — connects request builder, SDK client, and stream parser."""

from collections.abc import AsyncIterator
from decimal import Decimal

import anthropic
from anthropic import AsyncAnthropic

from agentkit._content import TextBlock
from agentkit._messages import Message, Usage
from agentkit.providers.anthropic.pricing import estimate_cost_usd
from agentkit.providers.anthropic.request_builder import build_anthropic_request
from agentkit.providers.anthropic.stream_parser import parse_anthropic_stream
from agentkit.providers.base import (
    ErrorEvent,
    Provider,
    ProviderCapabilities,
    ProviderEvent,
    ProviderRequest,
)


def _map_anthropic_error(exc: BaseException) -> ErrorEvent:  # noqa: PLR0911 — exhaustive SDK exception mapping
    """Map an Anthropic SDK exception to a provider ErrorEvent.

    The mapped ``code`` is consumed by StreamMux to pick a top-level ``ErrorCode``;
    the ``message`` is forwarded verbatim to the consumer for diagnostics.
    """
    if isinstance(exc, anthropic.AuthenticationError):
        return ErrorEvent(code="auth_failed", message=str(exc), recoverable=False)
    if isinstance(exc, anthropic.NotFoundError):
        return ErrorEvent(code="not_found", message=str(exc), recoverable=False)
    if isinstance(exc, anthropic.BadRequestError):
        return ErrorEvent(code="bad_request", message=str(exc), recoverable=False)
    if isinstance(exc, anthropic.RateLimitError):
        return ErrorEvent(code="rate_limited", message=str(exc), recoverable=True)
    if isinstance(exc, anthropic.APITimeoutError):
        return ErrorEvent(code="timeout", message=str(exc), recoverable=True)
    if isinstance(exc, anthropic.APIConnectionError):
        return ErrorEvent(code="connection", message=str(exc), recoverable=True)
    if isinstance(exc, anthropic.APIError):
        return ErrorEvent(code="provider_error", message=str(exc), recoverable=False)
    return ErrorEvent(
        code="provider_error", message=f"{type(exc).__name__}: {exc}", recoverable=False
    )


class AnthropicProvider(Provider):
    name = "anthropic"
    capabilities = ProviderCapabilities(
        supports_tool_use=True,
        supports_parallel_tools=True,
        supports_prompt_caching=True,
        supports_vision=True,
        supports_thinking=True,
        max_context_tokens=200_000,
        max_output_tokens=8_192,
    )

    def __init__(self, api_key: str | None = None, *, client: AsyncAnthropic | None = None) -> None:
        self._client = client or AsyncAnthropic(api_key=api_key)

    async def stream(self, request: ProviderRequest) -> AsyncIterator[ProviderEvent]:
        payload = build_anthropic_request(request)
        try:
            async with self._client.messages.stream(**payload) as stream:
                async for ev in parse_anthropic_stream(stream):
                    yield ev
        except Exception as exc:
            yield _map_anthropic_error(exc)

    def estimate_tokens(self, messages: list[Message]) -> int:
        # Rough token estimate — adapter could call Anthropic's token-count API
        # for more accuracy, but that costs an extra round-trip per turn.
        # Using isinstance(b, TextBlock) instead of hasattr(b, "text") to avoid
        # accidentally counting ThinkingBlock.text (which also has a .text field).
        total = 0
        for m in messages:
            for b in m.content:
                if isinstance(b, TextBlock):
                    total += len(b.text) // 4
        return total

    def estimate_cost(self, usage: Usage) -> Decimal:
        # Provider doesn't know which model — caller (TurnMetrics) must pass.
        # We expose the per-model helper for callers; this method is a default 0.
        return Decimal("0")

    def estimate_cost_for(self, model: str, usage: Usage) -> Decimal:
        return estimate_cost_usd(model, usage)
