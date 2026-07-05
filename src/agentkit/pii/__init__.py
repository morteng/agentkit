"""PII firewall subsystem.

Inert unless a consumer supplies a ``Detector`` + ``TokenMap`` (via
``wrap_provider``'s ``tmap_resolver``). agentkit owns the mechanism; the
consumer owns the recognizers, the durable token map, and the consent gate.
"""

from agentkit.pii.audit import AuditSink, OutboundAudit, emit_audit, set_audit_sink
from agentkit.pii.firewall import (
    Firewall,
    RehydrationRefused,
    ResidualTokenError,
)
from agentkit.pii.policy import PiiPolicy, ZdrRouteUnavailable
from agentkit.pii.protocols import Detector, TokenMap
from agentkit.pii.provider import (
    ScrubbingProvider,
    TokenMapResolver,
    wrap_provider,
)
from agentkit.pii.types import Action, RehydratePolicy, Span

__all__ = [
    "Action",
    "AuditSink",
    "Detector",
    "Firewall",
    "OutboundAudit",
    "PiiPolicy",
    "RehydratePolicy",
    "RehydrationRefused",
    "ResidualTokenError",
    "ScrubbingProvider",
    "Span",
    "TokenMap",
    "TokenMapResolver",
    "ZdrRouteUnavailable",
    "emit_audit",
    "set_audit_sink",
    "wrap_provider",
]
