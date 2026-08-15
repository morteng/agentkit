"""PII firewall subsystem.

Two halves, with different owners.

**Identity** — national ID numbers, bank accounts, names, addresses — is
domain knowledge, so agentkit ships no recognizers for it. The consumer
supplies a ``Detector`` + ``TokenMap`` (via ``wrap_provider``'s
``tmap_resolver``); without them the firewall is inert and costs nothing.

**Credentials** are not domain knowledge — a generated password looks the same
in every domain — so agentkit does ship ``SecretDetector``, and it is on by
default through ``CompositeDetector.with_defaults``. Compose it with your own
recognizers rather than replacing the detector wholesale::

    firewall = Firewall(
        CompositeDetector.with_defaults(MyDomainDetector()),
        PiiPolicy(),
    )

agentkit owns the mechanism; the consumer owns the domain recognizers, the
durable token map, and the consent gate.
"""

from agentkit.pii.audit import AuditSink, OutboundAudit, emit_audit, set_audit_sink
from agentkit.pii.composite import CompositeDetector, default_detector, default_detectors
from agentkit.pii.firewall import (
    Firewall,
    RehydrationRefused,
    ResidualTokenError,
)
from agentkit.pii.policy import BlockedModelError, PiiPolicy, ZdrRouteUnavailable
from agentkit.pii.protocols import Detector, FieldContextDetector, TokenMap
from agentkit.pii.provider import (
    ScrubbingProvider,
    TokenMapResolver,
    wrap_provider,
)
from agentkit.pii.secrets import (
    SECRET_CONTEXT_KEYWORDS,
    SECRET_FIELD_NAMES,
    SecretDetector,
    SecretPolicy,
    shannon_entropy,
)
from agentkit.pii.spans import merge_spans
from agentkit.pii.types import Action, RehydratePolicy, Span

__all__ = [
    "SECRET_CONTEXT_KEYWORDS",
    "SECRET_FIELD_NAMES",
    "Action",
    "AuditSink",
    "BlockedModelError",
    "CompositeDetector",
    "Detector",
    "FieldContextDetector",
    "Firewall",
    "OutboundAudit",
    "PiiPolicy",
    "RehydratePolicy",
    "RehydrationRefused",
    "ResidualTokenError",
    "ScrubbingProvider",
    "SecretDetector",
    "SecretPolicy",
    "Span",
    "TokenMap",
    "TokenMapResolver",
    "ZdrRouteUnavailable",
    "default_detector",
    "default_detectors",
    "emit_audit",
    "merge_spans",
    "set_audit_sink",
    "shannon_entropy",
    "wrap_provider",
]
