"""Consumer-supplied protocols: ``Detector`` and ``TokenMap``.

agentkit owns the mechanism (the ``Firewall`` and the decorating provider); the
consumer (REDACTED2) implements these two protocols with its Norwegian
recognizers and its ``ContactInfo``-backed deterministic map. REDACTED/REDACTED do
not implement them at all — they pass ``tmap_resolver`` returning ``None`` and
the firewall is inert.

agentkit does ship one detector of its own — ``SecretDetector``, which finds
credentials rather than identities. Compose it with your recognizers via
``CompositeDetector`` rather than replacing it: the ``Detector`` slot on
``Firewall`` is singular, so handing it a bare domain detector silently turns
everything else off.
"""

from typing import Protocol, runtime_checkable

from agentkit.pii.types import Span


@runtime_checkable
class Detector(Protocol):
    """Find PII spans in text. Fail-closed → ``TOKENIZE`` on uncertainty."""

    def detect(self, text: str) -> list[Span]: ...


@runtime_checkable
class FieldContextDetector(Protocol):
    """Optional ``Detector`` extension: detection that knows its field name.

    Structured surfaces — tool arguments, JSON tool results — carry the name of
    the field a string sits under, and that name is often the only evidence
    there is: ``{"password": "sommer2026"}`` has no detectable shape, only a
    label. The firewall calls ``detect_in_field`` on detectors that implement
    it and plain ``detect`` on those that do not, so the extension is purely
    additive — existing consumer detectors keep working untouched.

    ``field`` is ``None`` for free text with no enclosing field.
    """

    def detect(self, text: str) -> list[Span]: ...

    def detect_in_field(self, text: str, field: str | None) -> list[Span]: ...


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
