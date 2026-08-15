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


def test_routing_prefs_sends_a_conservative_default_without_zdr():
    """No ZDR requirement is not the same as no opinion.

    ``routing_prefs`` is only consulted for a request the firewall is *active*
    on — one carrying PII. Sending no preference there does not mean "no
    preference" on the wire; it means the provider's default applies, and
    provider defaults permit retention and training. So the firewall states
    ``data_collection="deny"`` explicitly, with fallbacks left on so routing
    degrades instead of failing.
    """
    fw = Firewall(FakeDetector(), PiiPolicy(require_zdr=False, eu_only=False))
    prefs = fw.routing_prefs()
    assert prefs is not None
    assert prefs.zdr is False
    assert prefs.data_collection == "deny"
    assert prefs.allow_fallbacks is True


def test_routing_prefs_default_data_collection_allow():
    fw = Firewall(FakeDetector(), PiiPolicy(require_zdr=False, default_data_collection="allow"))
    prefs = fw.routing_prefs()
    assert prefs is not None
    assert prefs.data_collection == "allow"
    assert prefs.allow_fallbacks is True


def test_routing_prefs_unset_restores_the_silent_payload():
    """The documented opt-out: emit nothing, exactly as before."""
    fw = Firewall(
        FakeDetector(),
        PiiPolicy(require_zdr=False, eu_only=False, default_data_collection="unset"),
    )
    assert fw.routing_prefs() is None


def test_routing_prefs_unset_still_carries_eu_only():
    fw = Firewall(
        FakeDetector(),
        PiiPolicy(require_zdr=False, eu_only=True, default_data_collection="unset"),
    )
    prefs = fw.routing_prefs()
    assert prefs is not None
    assert prefs.eu_only is True
    assert prefs.allow_fallbacks is True


def test_routing_prefs_eu_only():
    fw = Firewall(FakeDetector(), PiiPolicy(require_zdr=False, eu_only=True))
    prefs = fw.routing_prefs()
    assert prefs is not None
    assert prefs.eu_only is True
    assert prefs.data_collection == "deny"


def test_routing_prefs_zdr_carries_eu_only():
    fw = Firewall(FakeDetector(), PiiPolicy(require_zdr=True, eu_only=True))
    prefs = fw.routing_prefs()
    assert prefs is not None
    assert prefs.zdr is True
    assert prefs.eu_only is True
    assert prefs.allow_fallbacks is False


def test_conservative_default_reaches_the_openrouter_payload():
    fw = Firewall(FakeDetector(), PiiPolicy(require_zdr=False))
    req = ProviderRequest(model="openai/gpt-5.5", routing=fw.routing_prefs())
    assert build_openrouter_request(req)["provider"] == {
        "zdr": False,
        "data_collection": "deny",
        "allow_fallbacks": True,
    }
