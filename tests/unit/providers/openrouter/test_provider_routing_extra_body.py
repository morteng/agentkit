"""Routing prefs must ride on ``extra_body``, and the call must be SDK-legal.

``provider`` is a real top-level field of OpenRouter's HTTP API, so
``build_openrouter_request`` is right to emit it. The openai SDK, however, has
no ``provider`` parameter and no ``**kwargs`` catch-all, so splatting the
payload into ``chat.completions.create()`` raised ``TypeError`` before a
request was ever sent.

Why it survived to a release: it only fired when the request carried routing
prefs, and ``Firewall.routing_prefs()`` returned ``None`` unless the turn
carried PII. 0.22.0 made it never return ``None``, at which point *every*
request through a firewall-wrapped provider raised — including every request
from a deployment that merely sets ``require_zdr=True``.

Why the existing tests did not catch it, which is the more useful lesson: the
fake client in this package is ``async def fake_create(**kwargs)``. It accepts
any keyword at all, so a payload the real SDK rejects sails through a mock.
:func:`test_the_call_is_legal_against_the_real_sdk_signature` closes that by
binding the captured kwargs to the actual installed signature — the assertion a
``**kwargs`` double cannot make on the SDK's behalf.
"""

import inspect
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from agentkit.providers.base import ProviderRequest, RoutingPreferences
from agentkit.providers.openrouter.adapter import OpenRouterProvider


def _make_capturing_client() -> tuple[MagicMock, dict[str, Any]]:
    captured: dict[str, Any] = {}

    async def empty_iter():
        if False:
            yield

    async def fake_create(**kwargs: Any):
        captured.update(kwargs)
        return empty_iter()

    client = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=fake_create)
    return client, captured


def _request() -> ProviderRequest:
    return ProviderRequest(
        model="deepseek/deepseek-v4-flash",
        routing=RoutingPreferences(zdr=True, data_collection="deny", allow_fallbacks=False),
    )


async def _capture() -> dict[str, Any]:
    client, captured = _make_capturing_client()
    provider = OpenRouterProvider(api_key="x", client=client)
    async for _ in provider.stream(_request()):
        pass
    return captured


@pytest.mark.asyncio
async def test_routing_prefs_land_in_extra_body_not_as_a_top_level_kwarg() -> None:
    captured = await _capture()

    assert "provider" not in captured, (
        "routing prefs were passed as a top-level kwarg; the openai SDK has no "
        "such parameter and the call raises TypeError before sending"
    )
    extra_body = captured.get("extra_body") or {}
    assert extra_body.get("provider") == {
        "data_collection": "deny",
        "allow_fallbacks": False,
        "zdr": True,
    }, f"routing prefs missing or reshaped in extra_body: {extra_body!r}"


@pytest.mark.asyncio
async def test_the_call_is_legal_against_the_real_sdk_signature() -> None:
    """Bind the captured kwargs to the installed SDK. This is the teeth.

    The other test asserts where the field went; this one asserts the SDK would
    have accepted the call. Before the fix it failed with the same TypeError a
    real deployment got, which is the property worth pinning — the assertion
    survives the SDK renaming or removing parameters, whereas a hand-written
    allowlist of legal kwargs would quietly rot.
    """
    from openai.resources.chat.completions import AsyncCompletions

    captured = await _capture()
    sig = inspect.signature(AsyncCompletions.create)
    # `self` is unbound here; the captured kwargs are all keyword-only anyway.
    sig.bind_partial(None, **captured)


@pytest.mark.asyncio
async def test_no_routing_means_no_provider_key_at_all() -> None:
    """An empty ``extra_body`` is passed as ``None``, not ``{}``.

    Worth pinning because the inert path is the common one for consumers with
    no PII firewall, and a stray empty object in the request body is the kind
    of difference that changes provider-side behaviour for no stated reason.
    """
    client, captured = _make_capturing_client()
    provider = OpenRouterProvider(api_key="x", client=client)
    async for _ in provider.stream(ProviderRequest(model="deepseek/deepseek-v4-flash")):
        pass

    assert "provider" not in captured
    assert captured.get("extra_body") is None
