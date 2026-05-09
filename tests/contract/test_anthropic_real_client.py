"""Real-SDK contract tests for AnthropicProvider.

Each test exercises one request-payload field by driving the real Anthropic SDK
client with a stubbed transport. If agentkit ever sends an unknown kwarg or
mis-serialises a field, the SDK raises before the request leaves agentkit —
the test catches the regression at PR time.

Note on error surfacing: the adapter wraps all SDK exceptions in an ErrorEvent
rather than re-raising, so the tests assert that no ErrorEvent is yielded —
that's how a SDK TypeError (unknown kwarg) surfaces to a caller.

Note on thinking config: AnthropicProvider does NOT expose a ``thinking`` kwarg
on its constructor. Thinking is configured via ProviderRequest.thinking
(a ThinkingConfig instance), which build_anthropic_request() translates to the
Anthropic SDK's ``{"type": "enabled", "budget_tokens": N}`` wire shape.
"""

import pytest

from agentkit.providers.anthropic.adapter import AnthropicProvider
from agentkit.providers.base import ErrorEvent, ProviderRequest, ThinkingConfig


@pytest.mark.asyncio
async def test_minimal_request_accepted_by_real_sdk(real_anthropic_client):
    """A bare ProviderRequest must be valid against the real Anthropic SDK schema."""
    provider = AnthropicProvider(api_key="stub", client=real_anthropic_client)
    req = ProviderRequest(model="claude-opus-4-7")

    events = [ev async for ev in provider.stream(req)]
    error_events = [ev for ev in events if isinstance(ev, ErrorEvent)]
    assert error_events == [], f"Unexpected error events: {error_events}"


@pytest.mark.asyncio
async def test_thinking_config_accepted_by_real_sdk(real_anthropic_client):
    """Thinking config on ProviderRequest must produce a valid Anthropic SDK payload.

    ThinkingConfig is set on ProviderRequest.thinking, not on the provider
    constructor — build_anthropic_request() translates it to the SDK wire shape.
    If the translated payload is malformed, the real SDK raises, which the adapter
    surfaces as an ErrorEvent.
    """
    provider = AnthropicProvider(api_key="stub", client=real_anthropic_client)
    req = ProviderRequest(
        model="claude-opus-4-7",
        thinking=ThinkingConfig(enabled=True, budget_tokens=2000),
    )

    events = [ev async for ev in provider.stream(req)]
    error_events = [ev for ev in events if isinstance(ev, ErrorEvent)]
    assert error_events == [], (
        f"Unexpected error events (thinking payload malformed?): {error_events}"
    )
