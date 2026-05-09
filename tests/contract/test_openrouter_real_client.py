"""Real-SDK contract tests for OpenRouterProvider.

Each test exercises one request-payload field by driving the real openai SDK
client with a stubbed transport. If agentkit ever sends an unknown kwarg or
mis-serializes a field, the SDK raises before the request leaves agentkit —
the test catches the v0.2.0-class regression at PR time.

Note on error surfacing: the adapter wraps all SDK exceptions in an ErrorEvent
rather than re-raising, so the tests assert that no ErrorEvent is yielded —
that's how a SDK TypeError (unknown kwarg) surfaces to a caller.
"""

import pytest

from agentkit.providers.base import ErrorEvent, ProviderRequest
from agentkit.providers.openrouter.adapter import OpenRouterProvider


@pytest.mark.asyncio
async def test_reasoning_via_extra_body_accepted_by_real_sdk(real_openai_client):
    """Regression: v0.2.0 sent reasoning as a top-level kwarg; real SDK rejected.
    v0.2.1 routes through extra_body."""
    provider = OpenRouterProvider(
        api_key="stub",
        client=real_openai_client,
        reasoning={"effort": "medium"},
    )
    req = ProviderRequest(model="deepseek/deepseek-v4-flash")

    # If reasoning leaked back to the top-level kwargs, the real SDK raises
    # TypeError — caught by the adapter and surfaced as an ErrorEvent.
    # Asserting no ErrorEvent confirms reasoning went through extra_body correctly.
    events = [ev async for ev in provider.stream(req)]
    error_events = [ev for ev in events if isinstance(ev, ErrorEvent)]
    assert error_events == [], (
        f"Unexpected error events (reasoning kwarg leaked to top-level?): {error_events}"
    )


@pytest.mark.asyncio
async def test_no_reasoning_is_clean(real_openai_client):
    """Provider without reasoning config sends no extra_body — real SDK accepts."""
    provider = OpenRouterProvider(api_key="stub", client=real_openai_client)
    req = ProviderRequest(model="openai/gpt-5")
    events = [ev async for ev in provider.stream(req)]
    error_events = [ev for ev in events if isinstance(ev, ErrorEvent)]
    assert error_events == [], f"Unexpected error events: {error_events}"
