"""Redis-backed CheckpointStore. Raw-bytes payloads with TTL."""

from agentkit._ids import CheckpointId
from agentkit.store.checkpoint import CheckpointPayload, CheckpointStore
from agentkit.store.redis.client import RedisClient


class RedisCheckpointStore(CheckpointStore):
    def __init__(self, client: RedisClient, *, ttl_seconds: int = 24 * 60 * 60) -> None:
        self._c = client
        self._ttl = ttl_seconds

    async def save(self, checkpoint_id: CheckpointId, payload: CheckpointPayload) -> None:
        await self._c.redis.set(  # type: ignore[no-untyped-call]
            self._c.keys.checkpoint(checkpoint_id),
            payload,
            ex=self._ttl,
        )

    async def load(self, checkpoint_id: CheckpointId) -> CheckpointPayload | None:
        return await self._c.redis.get(self._c.keys.checkpoint(checkpoint_id))  # type: ignore[no-untyped-call]

    async def delete(self, checkpoint_id: CheckpointId) -> None:
        await self._c.redis.delete(self._c.keys.checkpoint(checkpoint_id))  # type: ignore[no-untyped-call]
