"""PiiPolicy.blocked_models is enforced at the egress boundary.

The field existed on the policy and nothing read it: a model the operator had
explicitly forbidden received PII anyway.
"""

import pytest

from agentkit._content import TextBlock
from agentkit._messages import MessageRole
from agentkit.pii.firewall import Firewall
from agentkit.pii.policy import BlockedModelError, PiiPolicy
from agentkit.pii.provider import wrap_provider
from agentkit.providers.base import ProviderRequest, TextDelta
from agentkit.providers.fakes import FakeProvider

from .conftest import FakeDetector, FakeTokenMap, make_msg

BLOCKED = "openai/gpt-5.5"


def _fw(detector: FakeDetector, **policy_kw: object) -> Firewall:
    return Firewall(detector=detector, policy=PiiPolicy(**policy_kw))  # type: ignore[arg-type]


def _req(model: str) -> ProviderRequest:
    return ProviderRequest(
        model=model,
        messages=[make_msg(MessageRole.USER, TextBlock(text="email kari@example.no"))],
    )


async def test_blocked_model_refuses_a_pii_request(detector: FakeDetector, tmap: FakeTokenMap):
    inner = FakeProvider().script(FakeProvider.text("should never run"))
    firewall = _fw(detector, blocked_models={BLOCKED})
    wrapped = wrap_provider(inner, firewall, tmap_resolver=lambda req: tmap)
    with pytest.raises(BlockedModelError, match=BLOCKED):
        _ = [ev async for ev in wrapped.stream(_req(BLOCKED))]


async def test_unblocked_model_still_streams(detector: FakeDetector, tmap: FakeTokenMap):
    inner = FakeProvider().script(FakeProvider.text("ok"))
    firewall = _fw(detector, require_zdr=False, blocked_models={BLOCKED})
    wrapped = wrap_provider(inner, firewall, tmap_resolver=lambda req: tmap)
    events = [ev async for ev in wrapped.stream(_req("anthropic/claude-sonnet-5"))]
    assert "".join(e.delta for e in events if isinstance(e, TextDelta)) == "ok"


async def test_inert_path_ignores_blocked_models(detector: FakeDetector):
    """No token map means no PII in this request, so the PII policy has no say.

    Blocking model availability outright is the caller's job, not the
    firewall's — and silently breaking non-PII traffic for consumers that run
    the firewall inert would be a surprise.
    """
    inner = FakeProvider().script(FakeProvider.text("ok"))
    firewall = _fw(detector, blocked_models={BLOCKED})
    wrapped = wrap_provider(inner, firewall, tmap_resolver=lambda req: None)
    events = [ev async for ev in wrapped.stream(_req(BLOCKED))]
    assert "".join(e.delta for e in events if isinstance(e, TextDelta)) == "ok"


async def test_empty_blocked_models_is_the_default(detector: FakeDetector, tmap: FakeTokenMap):
    inner = FakeProvider().script(FakeProvider.text("ok"))
    wrapped = wrap_provider(inner, _fw(detector, require_zdr=False), lambda req: tmap)
    events = [ev async for ev in wrapped.stream(_req(BLOCKED))]
    assert "".join(e.delta for e in events if isinstance(e, TextDelta)) == "ok"
