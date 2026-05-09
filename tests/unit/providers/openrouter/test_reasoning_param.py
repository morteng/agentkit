"""OpenRouterProvider forwards a reasoning config into the request payload.

OpenRouter's chat-completions API accepts a top-level ``reasoning`` object on
reasoning-capable models (DeepSeek-R1, GPT-5 reasoning, Claude extended
thinking, etc.). The provider exposes this as a constructor option so a single
provider instance carries one model's deployment policy.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from agentkit.providers.base import ProviderRequest
from agentkit.providers.openrouter.adapter import OpenRouterProvider


def _make_capturing_client() -> tuple[MagicMock, dict]:
    """Build a fake AsyncOpenAI whose ``chat.completions.create`` records kwargs.

    Returns the client plus the dict that the captured kwargs land in. The
    create() coroutine resolves to an empty async iterator so the provider's
    stream parser yields no events.
    """
    captured: dict = {}

    async def empty_iter():
        if False:
            yield  # never executes; declares this as an async generator

    async def fake_create(**kwargs):
        captured.update(kwargs)
        return empty_iter()

    client = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=fake_create)
    return client, captured


@pytest.mark.asyncio
async def test_reasoning_param_added_to_payload_when_set():
    client, captured = _make_capturing_client()
    provider = OpenRouterProvider(
        api_key="x",
        client=client,
        reasoning={"effort": "medium"},
    )
    req = ProviderRequest(model="deepseek/deepseek-v4-flash")

    async for _ in provider.stream(req):
        pass

    assert captured.get("reasoning") == {"effort": "medium"}


@pytest.mark.asyncio
async def test_reasoning_param_omitted_when_unset():
    client, captured = _make_capturing_client()
    provider = OpenRouterProvider(api_key="x", client=client)
    req = ProviderRequest(model="deepseek/deepseek-v4-flash")

    async for _ in provider.stream(req):
        pass

    assert "reasoning" not in captured


@pytest.mark.asyncio
async def test_reasoning_param_supports_max_tokens_shape():
    """The provider forwards the dict verbatim — Anthropic-style max_tokens works too."""
    client, captured = _make_capturing_client()
    provider = OpenRouterProvider(
        api_key="x",
        client=client,
        reasoning={"max_tokens": 2000, "exclude": True},
    )
    req = ProviderRequest(model="anthropic/claude-opus-4-7")

    async for _ in provider.stream(req):
        pass

    assert captured.get("reasoning") == {"max_tokens": 2000, "exclude": True}
