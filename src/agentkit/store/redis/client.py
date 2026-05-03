"""Shared Redis connection-pool wrapper.

The async ``redis.asyncio`` client supports connection pooling out of the box;
this thin wrapper standardises construction and adds the KeyBuilder.
"""

from dataclasses import dataclass

from redis.asyncio import ConnectionPool, Redis

from agentkit.store.redis.keys import KeyBuilder


@dataclass(frozen=True)
class RedisStoreConfig:
    url: str = "redis://localhost:6379"
    prefix: str = "agentkit"
    max_connections: int = 50
    decode_responses: bool = False


class RedisClient:
    """Wraps a Redis connection pool plus the KeyBuilder for the deployment."""

    def __init__(self, config: RedisStoreConfig) -> None:
        self._config = config
        self._pool = ConnectionPool.from_url(  # type: ignore[reportUnknownMemberType]
            config.url,
            max_connections=config.max_connections,
            decode_responses=config.decode_responses,
        )
        self._redis = Redis(connection_pool=self._pool)
        self.keys = KeyBuilder(prefix=config.prefix)

    @property
    def redis(self) -> Redis:
        return self._redis

    async def close(self) -> None:
        await self._redis.aclose()
        await self._pool.disconnect()
