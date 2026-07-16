"""RoutingPreferences → request-builder provider block; routing_prefs derivation."""

from agentkit.pii.firewall import Firewall
from agentkit.pii.policy import PiiPolicy
from agentkit.providers.anthropic.request_builder import build_anthropic_request
from agentkit.providers.base import ProviderRequest, RoutingPreferences
from agentkit.providers.openrouter.request_builder import build_openrouter_request

from .conftest import FakeDetector


def test_openrouter_omits_provider_block_when_routing_none():
    req = ProviderRequest(model="openai/gpt-5.5")
    payload = build_openrouter_request(req)
    assert "provider" not in payload


def test_openrouter_emits_provider_block_when_routing_set():
    req = ProviderRequest(
        model="openai/gpt-5.5",
        routing=RoutingPreferences(zdr=True, data_collection="deny", allow_fallbacks=False),
    )
    payload = build_openrouter_request(req)
    assert payload["provider"] == {
        "zdr": True,
        "data_collection": "deny",
        "allow_fallbacks": False,
    }


def test_openrouter_eu_only_adds_only_list():
    req = ProviderRequest(
        model="openai/gpt-5.5",
        routing=RoutingPreferences(
            zdr=True,
            data_collection="deny",
            allow_fallbacks=False,
            eu_only=True,
            only=["mistral"],
        ),
    )
    payload = build_openrouter_request(req)
    assert payload["provider"]["only"] == ["mistral"]


def test_openrouter_eu_only_without_only_list_omits_only():
    req = ProviderRequest(
        model="openai/gpt-5.5",
        routing=RoutingPreferences(eu_only=True),
    )
    payload = build_openrouter_request(req)
    assert "only" not in payload["provider"]


def test_anthropic_builder_accepts_routing_without_breaking():
    req = ProviderRequest(
        model="anthropic/claude-sonnet-5",
        routing=RoutingPreferences(zdr=True),
    )
    payload = build_anthropic_request(req)
    # Anthropic has no per-request ZDR field — payload must not carry a
    # fabricated provider block, but must still build.
    assert "provider" not in payload
    assert payload["model"] == "anthropic/claude-sonnet-5"


def test_routing_prefs_require_zdr():
    fw = Firewall(FakeDetector(), PiiPolicy(require_zdr=True))
    prefs = fw.routing_prefs()
    assert prefs is not None
    assert prefs.zdr is True
    assert prefs.data_collection == "deny"
    assert prefs.allow_fallbacks is False


def test_routing_prefs_none_when_no_policy():
    fw = Firewall(FakeDetector(), PiiPolicy(require_zdr=False, eu_only=False))
    assert fw.routing_prefs() is None


def test_routing_prefs_eu_only():
    fw = Firewall(FakeDetector(), PiiPolicy(require_zdr=False, eu_only=True))
    prefs = fw.routing_prefs()
    assert prefs is not None
    assert prefs.eu_only is True
