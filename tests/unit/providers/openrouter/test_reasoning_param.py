"""OpenRouterProvider forwards a reasoning config via ``extra_body``.

OpenRouter's chat-completions API accepts a top-level ``reasoning`` object on
reasoning-capable models (DeepSeek-R1, GPT-5 reasoning, Claude extended
thinking, etc.). The openai SDK validates kwargs against the OpenAI schema and
rejects unknown fields at the call site, so OpenRouter passthroughs must ride
on ``extra_body`` rather than as top-level kwargs. The provider exposes the
config as a constructor option so a single provider instance carries one
model's deployment policy.
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
async def test_reasoning_param_lands_in_extra_body_not_payload():
    """Passthrough fields must ride on ``extra_body``; top-level kwargs would be
    rejected by the real openai SDK with TypeError."""
    client, captured = _make_capturing_client()
    provider = OpenRouterProvider(
        api_key="x",
        client=client,
        reasoning={"effort": "medium"},
    )
    req = ProviderRequest(model="deepseek/deepseek-v4-flash")

    async for _ in provider.stream(req):
        pass

    # Goes through extra_body, not as a top-level kwarg.
    assert captured.get("extra_body") == {"reasoning": {"effort": "medium"}}
    assert "reasoning" not in captured


@pytest.mark.asyncio
async def test_reasoning_param_omitted_when_unset():
    """No reasoning config -> ``extra_body`` is None (not an empty dict), so
    the openai SDK doesn't send a stray empty body field."""
    client, captured = _make_capturing_client()
    provider = OpenRouterProvider(api_key="x", client=client)
    req = ProviderRequest(model="deepseek/deepseek-v4-flash")

    async for _ in provider.stream(req):
        pass

    assert "reasoning" not in captured
    assert captured.get("extra_body") is None


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

    assert captured.get("extra_body") == {"reasoning": {"max_tokens": 2000, "exclude": True}}


@pytest.mark.asyncio
async def test_real_openai_client_rejects_unknown_kwargs():
    """Regression: the real AsyncOpenAI client raises TypeError on unknown kwargs.

    This is the failure that hit production in v0.112.9 — confirming the SDK
    contract here documents *why* extra_body is mandatory for passthroughs.
    """
    from openai import AsyncOpenAI

    real_client = AsyncOpenAI(api_key="x", base_url="http://localhost:1")
    with pytest.raises(TypeError, match="reasoning"):
        await real_client.chat.completions.create(  # type: ignore[reportUnknownArgumentType]
            model="deepseek/deepseek-v4-flash",
            messages=[{"role": "user", "content": "hi"}],
            reasoning={"effort": "medium"},  # type: ignore[arg-type]
        )
