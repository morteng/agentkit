"""Redis-backed MemoryStore (keyword-search variant).

Stores values as JSON at scoped keys. Keeps a per-scope SET of keys so
``list_keys`` is O(N) without SCAN. Search is naive substring matching over
``MemoryValue.text`` — fine for the default; vector search is opt-in.
"""

from agentkit.store.memory import MemoryHit, MemoryScope, MemoryStore, MemoryValue
from agentkit.store.redis.client import RedisClient
from agentkit.store.redis.serialization import from_versioned_json, to_versioned_json

_SCHEMA_V = 1


class RedisMemoryStore(MemoryStore):
    def __init__(self, client: RedisClient) -> None:
        self._c = client

    async def save(self, scope: MemoryScope, key: str, value: MemoryValue) -> None:
        await self._c.redis.set(  # type: ignore[no-untyped-call]
            self._c.keys.memory(scope, key),
            to_versioned_json(value.model_dump(mode="json"), schema_version=_SCHEMA_V),
        )
        await self._c.redis.sadd(self._c.keys.memory_index(scope), key)  # type: ignore[no-untyped-call]

    async def recall(self, scope: MemoryScope, key: str) -> MemoryValue | None:
        raw = await self._c.redis.get(self._c.keys.memory(scope, key))  # type: ignore[no-untyped-call]
        if raw is None:
            return None
        assert isinstance(raw, bytes)
        data, _ = from_versioned_json(raw)
        return MemoryValue.model_validate(data)

    async def search(
        self,
        scope: MemoryScope,
        query: str,
        *,
        limit: int = 10,
    ) -> list[MemoryHit]:
        ql = query.lower()
        hits: list[MemoryHit] = []
        for key in await self.list_keys(scope):
            value = await self.recall(scope, key)
            if value is None:
                continue
            if ql in value.text.lower():
                hits.append(MemoryHit(key=key, value=value, score=1.0))
            if len(hits) >= limit:
                break
        return hits

    async def list_keys(self, scope: MemoryScope) -> list[str]:
        raws = await self._c.redis.smembers(self._c.keys.memory_index(scope))  # type: ignore[no-untyped-call]
        decoded: list[str] = [r.decode() if isinstance(r, bytes) else r for r in raws]  # type: ignore[union-attr]
        return sorted(decoded)

    async def delete(self, scope: MemoryScope, key: str) -> None:
        await self._c.redis.delete(self._c.keys.memory(scope, key))  # type: ignore[no-untyped-call]
        await self._c.redis.srem(self._c.keys.memory_index(scope), key)  # type: ignore[no-untyped-call]
