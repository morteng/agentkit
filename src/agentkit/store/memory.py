"""Memory store protocol + scope/value/hit types."""

from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field


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


class MemoryHit(BaseModel):
    key: str
    value: MemoryValue
    score: float  # search ranking; 1.0 == exact, 0..1 fuzzy


@runtime_checkable
class MemoryStore(Protocol):
    """Long-lived facts the agent extracts and recalls."""

    async def save(self, scope: MemoryScope, key: str, value: MemoryValue) -> None: ...

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
