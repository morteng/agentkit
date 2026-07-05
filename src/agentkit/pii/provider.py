"""``wrap_provider`` — the decorating Provider that enforces the firewall.

This is the single egress choke-point (§2). It scrubs the whole request on the
way out and passes provider events through unchanged on the way back — text
deltas and tool-call args carry placeholders by design (the consumer's SSE tap
rehydrates for display; tools get placeholders via the DENY default). Only
``ErrorEvent.message`` is scrubbed, so PII in errors never reaches logs/ledger.
"""

from collections.abc import AsyncIterator, Callable
from decimal import Decimal

from agentkit._messages import Message, Usage
from agentkit.pii.audit import emit_audit
from agentkit.pii.firewall import Firewall
from agentkit.pii.policy import ZdrRouteUnavailable
from agentkit.pii.protocols import TokenMap
from agentkit.providers.base import (
    ErrorEvent,
    Provider,
    ProviderCapabilities,
    ProviderEvent,
    ProviderRequest,
)

TokenMapResolver = Callable[[ProviderRequest], TokenMap | None]

#: Error codes / substrings that signal "no compliant provider could be routed"
#: — the fail-closed trigger when ``require_zdr`` is set.
_ROUTE_FAILURE_MARKERS = (
    "no_compliant_provider",
    "no allowed providers",
    "no endpoints found",
    "no_endpoints",
    "zdr",
    "data policy",
    "data_policy",
    "no provider",
)


def _looks_like_route_failure(event: ErrorEvent) -> bool:
    haystack = f"{event.code} {event.message}".lower()
    return any(marker in haystack for marker in _ROUTE_FAILURE_MARKERS)


class ScrubbingProvider(Provider):
    """A Provider decorator applying a ``Firewall`` at the egress boundary.

    Inert when ``tmap_resolver`` returns ``None`` for a request (REDACTED/REDACTED
    pay nothing): the request and every event pass through untouched.
    """

    name: str
    capabilities: ProviderCapabilities

    def __init__(
        self,
        inner: Provider,
        firewall: Firewall,
        tmap_resolver: TokenMapResolver,
    ) -> None:
        self._inner = inner
        self._firewall = firewall
        self._resolve = tmap_resolver
        # Delegate identity/capabilities to the wrapped provider (inner is fixed).
        self.name = inner.name
        self.capabilities = inner.capabilities

    def estimate_tokens(self, messages: list[Message]) -> int:
        return self._inner.estimate_tokens(messages)

    def estimate_cost(self, usage: Usage) -> Decimal:
        return self._inner.estimate_cost(usage)

    async def stream(self, request: ProviderRequest) -> AsyncIterator[ProviderEvent]:
        tmap = self._resolve(request)
        if tmap is None:
            # INERT — no candidate context; pass through untouched.
            async for event in self._inner.stream(request):
                yield event
            return

        scrubbed = self._firewall.scrub_request(request, tmap)
        routing = self._firewall.routing_prefs()
        if routing is not None:
            scrubbed.routing = routing

        emit_audit(self._firewall.build_audit(scrubbed))

        require_zdr = self._firewall.policy.require_zdr
        async for event in self._inner.stream(scrubbed):
            if isinstance(event, ErrorEvent):
                if require_zdr and _looks_like_route_failure(event):
                    raise ZdrRouteUnavailable(
                        f"no zero-data-retention route for PII-flagged call: "
                        f"{event.code}: {event.message}"
                    )
                # Scrub PII out of the error before it hits logs/ledger.
                yield event.model_copy(
                    update={"message": self._firewall.scrub_text(event.message, tmap)}
                )
            else:
                # Text deltas and tool-call args keep placeholders by design.
                yield event


def wrap_provider(
    inner: Provider,
    firewall: Firewall,
    tmap_resolver: TokenMapResolver,
) -> Provider:
    """Wrap ``inner`` in a firewall-enforcing decorator.

    ``tmap_resolver(request) -> TokenMap | None``: return the per-candidate
    token map for a request, or ``None`` to make the firewall inert for that
    request (no PII context).
    """
    return ScrubbingProvider(inner, firewall, tmap_resolver)
