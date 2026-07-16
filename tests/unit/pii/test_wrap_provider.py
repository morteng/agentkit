"""wrap_provider: INERT passthrough, scrubbing path, ZDR fail-closed, error scrub."""

import pytest

from agentkit._content import TextBlock
from agentkit._messages import MessageRole
from agentkit.pii.firewall import Firewall
from agentkit.pii.policy import PiiPolicy, ZdrRouteUnavailable
from agentkit.pii.provider import wrap_provider
from agentkit.providers.base import ErrorEvent, ProviderRequest, TextDelta
from agentkit.providers.fakes import FakeProvider

from .conftest import FakeDetector, FakeTokenMap, make_msg


def _fw(detector: FakeDetector, **policy_kw: object) -> Firewall:
    return Firewall(detector=detector, policy=PiiPolicy(**policy_kw))  # type: ignore[arg-type]


def _req(text: str) -> ProviderRequest:
    return ProviderRequest(
        model="openai/gpt-5.5",
        messages=[make_msg(MessageRole.USER, TextBlock(text=text))],
    )


async def _drain(agen):  # type: ignore[no-untyped-def]
    return [ev async for ev in agen]


async def test_inert_when_resolver_returns_none(detector: FakeDetector):
    inner = FakeProvider().script(FakeProvider.text("Kari Nordmann stays raw"))
    wrapped = wrap_provider(inner, _fw(detector), tmap_resolver=lambda req: None)
    req = _req("Kari Nordmann kari@example.no")
    events = await _drain(wrapped.stream(req))
    text = "".join(e.delta for e in events if isinstance(e, TextDelta))
    # Passthrough: reply is byte-identical, request not scrubbed.
    assert text == "Kari Nordmann stays raw"


async def test_inert_delegates_metadata(detector: FakeDetector):
    inner = FakeProvider()
    wrapped = wrap_provider(inner, _fw(detector), tmap_resolver=lambda req: None)
    assert wrapped.name == inner.name
    assert wrapped.capabilities == inner.capabilities


async def test_scrubbing_path_scrubs_request(detector: FakeDetector, tmap: FakeTokenMap):
    captured: dict[str, ProviderRequest] = {}

    class CapturingFake(FakeProvider):
        async def stream(self, request):  # type: ignore[no-untyped-def, override]
            captured["req"] = request
            async for ev in super().stream(request):
                yield ev

    inner = CapturingFake().script(FakeProvider.text("ok"))
    wrapped = wrap_provider(inner, _fw(detector), tmap_resolver=lambda req: tmap)
    await _drain(wrapped.stream(_req("email kari@example.no now")))
    sent = captured["req"]
    assert "kari@example.no" not in sent.messages[0].content[0].text  # type: ignore[union-attr]
    # Routing attached (require_zdr default True).
    assert sent.routing is not None
    assert sent.routing.zdr is True
    assert sent.routing.allow_fallbacks is False


async def test_error_message_is_scrubbed(detector: FakeDetector, tmap: FakeTokenMap):
    inner = FakeProvider().script(FakeProvider.error("server_error", "failed for kari@example.no"))
    # require_zdr False so a generic error is not treated as a route failure.
    wrapped = wrap_provider(inner, _fw(detector, require_zdr=False), tmap_resolver=lambda req: tmap)
    events = await _drain(wrapped.stream(_req("hi")))
    err = next(e for e in events if isinstance(e, ErrorEvent))
    assert "kari@example.no" not in err.message
    assert "[EMAIL_1]" in err.message


async def test_zdr_route_failure_raises(detector: FakeDetector, tmap: FakeTokenMap):
    inner = FakeProvider().script(
        FakeProvider.error("no_compliant_provider", "no ZDR endpoints found")
    )
    wrapped = wrap_provider(inner, _fw(detector), tmap_resolver=lambda req: tmap)
    with pytest.raises(ZdrRouteUnavailable):
        await _drain(wrapped.stream(_req("hi")))


async def test_zdr_upstream_org_disabled_raises(detector: FakeDetector, tmap: FakeTokenMap):
    # Observed live: a ZDR-capable upstream (Amazon Bedrock) rejects the routed
    # request with "This organization has been disabled". Under require_zdr this
    # is a route failure — refuse, don't surface a generic error.
    inner = FakeProvider().script(
        FakeProvider.error(
            "provider_error",
            "Provider returned error: This organization has been disabled.",
        )
    )
    wrapped = wrap_provider(inner, _fw(detector), tmap_resolver=lambda req: tmap)
    with pytest.raises(ZdrRouteUnavailable):
        await _drain(wrapped.stream(_req("hi")))
