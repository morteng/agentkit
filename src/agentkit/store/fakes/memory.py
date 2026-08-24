"""In-memory FakeMemoryStore — naive substring search."""

from agentkit._content import Provenance
from agentkit.store.memory import (
    MemoryHit,
    MemoryScope,
    MemoryStore,
    MemoryValue,
    stamp_provenance,
)


class FakeMemoryStore(MemoryStore):
    def __init__(self) -> None:
        self._data: dict[MemoryScope, dict[str, MemoryValue]] = {}

    async def save(
        self,
        scope: MemoryScope,
        key: str,
        value: MemoryValue,
        *,
        provenance: Provenance | None = None,
    ) -> None:
        # Stamped rather than ignored: the fake is what consumers test their
        # own memory behaviour against, so a fake that drops provenance would
        # certify a laundering bug as fixed.
        self._data.setdefault(scope, {})[key] = stamp_provenance(value, provenance)

    async def recall(self, scope: MemoryScope, key: str) -> MemoryValue | None:
        return self._data.get(scope, {}).get(key)

    async def search(
        self,
        scope: MemoryScope,
        query: str,
        *,
        limit: int = 10,
    ) -> list[MemoryHit]:
        ql = query.lower()
        hits: list[MemoryHit] = []
        for key, value in self._data.get(scope, {}).items():
            if ql in value.text.lower():
                hits.append(MemoryHit(key=key, value=value, score=1.0))
        return hits[:limit]

    async def list_keys(self, scope: MemoryScope) -> list[str]:
        return list(self._data.get(scope, {}).keys())

    async def delete(self, scope: MemoryScope, key: str) -> None:
        self._data.get(scope, {}).pop(key, None)
