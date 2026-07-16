"""Consumer-supplied protocols: ``Detector`` and ``TokenMap``.

agentkit owns the mechanism (the ``Firewall`` and the decorating provider); the
consumer (REDACTED2) implements these two protocols with its Norwegian
recognizers and its ``ContactInfo``-backed deterministic map. REDACTED/REDACTED do
not implement them at all — they pass ``tmap_resolver`` returning ``None`` and
the firewall is inert.
"""

from typing import Protocol, runtime_checkable

from agentkit.pii.types import Span


@runtime_checkable
class Detector(Protocol):
    """Find PII spans in text. Fail-closed → ``TOKENIZE`` on uncertainty."""

    def detect(self, text: str) -> list[Span]: ...


@runtime_checkable
class TokenMap(Protocol):
    """Durable, per-candidate, deterministic value↔token map.

    ``token_for`` is stable: the same ``(value)`` maps to the same token
    forever, so a stored+rehydrated artifact re-tokenizes correctly when it
    flows back as tool context in a later session.
    """

    def token_for(self, value: str, kind: str) -> str: ...

    def value_for(self, token: str) -> str | None: ...

    def all_tokens(self) -> set[str]: ...
