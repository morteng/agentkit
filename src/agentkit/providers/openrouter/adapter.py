"""OpenRouterProvider — uses the OpenAI SDK pointed at OpenRouter's endpoint."""

from collections.abc import AsyncIterator
from decimal import Decimal
from typing import Any

import openai
from openai import AsyncOpenAI

from agentkit._content import TextBlock
from agentkit._messages import Message, Usage
from agentkit.providers.base import (
    ErrorEvent,
    Provider,
    ProviderCapabilities,
    ProviderEvent,
    ProviderRequest,
)
from agentkit.providers.openrouter.pricing import estimate_cost_usd
from agentkit.providers.openrouter.request_builder import build_openrouter_request
from agentkit.providers.openrouter.stream_parser import parse_openrouter_stream


def _map_openai_error(exc: BaseException) -> ErrorEvent:  # noqa: PLR0911 — exhaustive SDK exception mapping
    """Map an OpenAI SDK exception to a provider ErrorEvent.

    Same mapping shape as the Anthropic adapter so consumers get a uniform error
    code surface regardless of provider.
    """
    if isinstance(exc, openai.AuthenticationError):
        return ErrorEvent(code="auth_failed", message=str(exc), recoverable=False)
    if isinstance(exc, openai.NotFoundError):
        return ErrorEvent(code="not_found", message=str(exc), recoverable=False)
    if isinstance(exc, openai.BadRequestError):
        return ErrorEvent(code="bad_request", message=str(exc), recoverable=False)
    if isinstance(exc, openai.RateLimitError):
        return ErrorEvent(code="rate_limited", message=str(exc), recoverable=True)
    if isinstance(exc, openai.APITimeoutError):
        return ErrorEvent(code="timeout", message=str(exc), recoverable=True)
    if isinstance(exc, openai.APIConnectionError):
        return ErrorEvent(code="connection", message=str(exc), recoverable=True)
    if isinstance(exc, openai.APIError):
        return ErrorEvent(code="provider_error", message=str(exc), recoverable=False)
    return ErrorEvent(
        code="provider_error", message=f"{type(exc).__name__}: {exc}", recoverable=False
    )


# Known per-model overrides. The default ``capabilities`` attribute is the
# conservative fallback; consumers needing accurate limits for a specific
# model should call :meth:`OpenRouterProvider.capabilities_for(model)` or
# pass ``model_capabilities`` to the constructor for new entries.
_MODEL_CAPABILITIES: dict[str, ProviderCapabilities] = {
    "deepseek/deepseek-v4-flash": ProviderCapabilities(
        supports_tool_use=True,
        supports_parallel_tools=True,
        supports_prompt_caching=True,
        supports_vision=False,
        supports_thinking=False,
        max_context_tokens=1_048_576,
        max_output_tokens=8_192,
    ),
    "deepseek/deepseek-v4-pro": ProviderCapabilities(
        supports_tool_use=True,
        supports_parallel_tools=True,
        supports_prompt_caching=True,
        supports_vision=False,
        supports_thinking=False,
        max_context_tokens=1_048_576,
        max_output_tokens=8_192,
    ),
    "openrouter/owl-alpha": ProviderCapabilities(
        supports_tool_use=True,
        supports_parallel_tools=True,
        supports_prompt_caching=False,
        supports_vision=False,
        supports_thinking=False,
        max_context_tokens=128_000,
        max_output_tokens=8_192,
    ),
}


class OpenRouterProvider(Provider):
    name = "openrouter"
    # Conservative class-level default. Consumers should prefer
    # :meth:`capabilities_for` when they know which model they're targeting.
    capabilities = ProviderCapabilities(
        supports_tool_use=True,
        supports_parallel_tools=True,
        supports_prompt_caching=True,
        supports_vision=True,
        supports_thinking=False,
        max_context_tokens=128_000,
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
        model_capabilities: dict[str, ProviderCapabilities] | None = None,
        reasoning: dict[str, Any] | None = None,
    ) -> None:
        self._client = client or AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._extra_headers: dict[str, str] = {}
        if http_referer:
            self._extra_headers["HTTP-Referer"] = http_referer
        if x_title:
            self._extra_headers["X-Title"] = x_title
        # Caller-supplied overrides win over the built-in table.
        self._model_capabilities = {**_MODEL_CAPABILITIES, **(model_capabilities or {})}
        # Per-provider-instance reasoning config. OpenRouter accepts a top-level
        # ``reasoning`` object on chat-completions requests for reasoning-capable
        # models (DeepSeek-R1, GPT-5 reasoning, Claude extended thinking, etc.).
        # Shape: {"effort": "xhigh"|"high"|"medium"|"low"|"minimal"|"none"} OR
        # {"max_tokens": int} OR {"enabled": bool} OR {"exclude": bool}. Forwarded
        # verbatim so consumers control the wire shape; ``None`` (the default)
        # omits the field entirely.
        self._reasoning = reasoning

    def capabilities_for(self, model: str) -> ProviderCapabilities:
        """Return capabilities for ``model`` if known, else the conservative default.

        Falls back to :attr:`capabilities` when the model isn't in the built-in
        table. Add overrides via the ``model_capabilities`` constructor argument
        to teach the provider about new models without forking the library.
        """
        return self._model_capabilities.get(model, self.capabilities)

    async def stream(self, request: ProviderRequest) -> AsyncIterator[ProviderEvent]:
        payload = build_openrouter_request(request)
        # Always include usage info per chunk via stream_options when supported.
        payload.setdefault("stream_options", {"include_usage": True})
        # Reasoning config is provider-instance-scoped (set in __init__) rather
        # than per-request because the typical use case is "always send effort=
        # medium for this DeepSeek-V4 deployment" — a model-level deployment
        # decision, not a per-turn one. Callers needing per-turn variation can
        # construct multiple providers or extend ProviderRequest in a future
        # revision.
        # The openai SDK validates kwargs against the OpenAI schema and rejects
        # unknown fields, so OpenRouter passthroughs like ``reasoning`` must go
        # through ``extra_body`` rather than as a top-level kwarg.
        extra_body: dict[str, Any] = {}
        if self._reasoning is not None:
            extra_body["reasoning"] = self._reasoning
        try:
            chunks = await self._client.chat.completions.create(  # type: ignore[reportUnknownVariableType]
                extra_headers=self._extra_headers or None,
                extra_body=extra_body or None,
                **payload,
            )
            async for ev in parse_openrouter_stream(chunks):  # type: ignore[reportUnknownArgumentType]
                yield ev
        except Exception as exc:
            yield _map_openai_error(exc)

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
