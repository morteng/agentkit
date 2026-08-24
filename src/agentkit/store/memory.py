"""Memory store protocol + scope/value/hit types.

PROVENANCE ON THE MEMORY PATH (added 2026-08-24)
------------------------------------------------
Recalled memories are injected as context at the start of a turn, which makes
this module a *persistence layer for context*. Until now a write dropped the
trust classification the taint system had already computed
(:class:`~agentkit._content.Provenance`), so ``save()`` followed by
``recall()`` in a later turn returned untrusted third-party text
indistinguishable from a fact the host wrote itself — a laundering path
straight through the anti-injection control in :mod:`agentkit.guards.taint`.

The classification now lives on :attr:`MemoryValue.provenance`, so it survives
the round trip and both read methods (``recall`` and ``search``, which returns
whole values inside :class:`MemoryHit`) surface it without a signature change.
``save()`` additionally accepts a keyword for callers that hold a value they
did not construct and want to label it at the write boundary.
"""

from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from agentkit._content import Provenance


class MemoryScope(BaseModel):
    """Hierarchical scope for memory isolation.

    Comparison and hashing make scopes usable as cache keys.
    `session_id=None` means persistent (cross-session) memory.
    """

    namespace: str
    tenant_id: str | None = None
    user_id: str | None = None
    session_id: str | None = None

    model_config = {"frozen": True}


class MemoryValue(BaseModel):
    """Value stored in MemoryStore. Free-form payload + indexable text."""

    text: str  # human-readable; what search() indexes
    payload: dict[str, Any] = Field(default_factory=dict)  # type: ignore[reportUnknownVariableType]
    tags: list[str] = Field(default_factory=list)  # type: ignore[reportUnknownVariableType]
    created_at: datetime
    updated_at: datetime
    provenance: Provenance = Provenance.SYSTEM
    """Trust level of :attr:`text` — where the remembered bytes came from.

    Defaults to ``SYSTEM``, so a value constructed before this field existed,
    and a payload deserialised from a store written before it existed, both
    keep the trusted treatment they had. A writer that is recording something
    it did not author — anything a model composed, anything derived from a
    tool result — must say so, or the read side has no way to tell.

    ``UNTRUSTED`` here taints the turn that recalls it, exactly as an untrusted
    tool result does; see :mod:`agentkit.guards.taint`.
    """


class MemoryHit(BaseModel):
    key: str
    value: MemoryValue
    score: float  # search ranking; 1.0 == exact, 0..1 fuzzy


def stamp_provenance(value: MemoryValue, provenance: Provenance | None) -> MemoryValue:
    """Apply ``save()``'s ``provenance`` keyword to the value being persisted.

    ``None`` — the protocol default — means *do not override*: the value keeps
    whatever :attr:`MemoryValue.provenance` it already carries. That is the
    only default that cannot itself launder, because a literal
    ``Provenance.SYSTEM`` default would silently overwrite a classification the
    caller had already put on the value.

    Every :class:`MemoryStore` implementation should route its write through
    this, so the keyword means the same thing in all of them.
    """
    if provenance is None or provenance is value.provenance:
        return value
    return value.model_copy(update={"provenance": provenance})


@runtime_checkable
class MemoryStore(Protocol):
    """Long-lived facts the agent extracts and recalls."""

    async def save(
        self,
        scope: MemoryScope,
        key: str,
        value: MemoryValue,
        *,
        provenance: Provenance | None = None,
    ) -> None:
        """Persist ``value`` at ``key``.

        ``provenance`` labels what is being written; ``None`` keeps the label
        already on ``value``. Implementations must persist the result of
        :func:`stamp_provenance` — a store that drops it re-opens the
        ``save()`` → ``recall()`` laundering path this keyword exists to close.

        Keyword-only with a default, so every implementation written before
        provenance existed remains structurally valid against this protocol.
        """
        ...

    async def recall(self, scope: MemoryScope, key: str) -> MemoryValue | None: ...

    async def search(
        self,
        scope: MemoryScope,
        query: str,
        *,
        limit: int = 10,
    ) -> list[MemoryHit]: ...

    async def list_keys(self, scope: MemoryScope) -> list[str]: ...

    async def delete(self, scope: MemoryScope, key: str) -> None: ...
