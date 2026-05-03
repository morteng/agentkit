import os
from collections.abc import Iterator

import pytest
from testcontainers.redis import RedisContainer

from agentkit.store.redis.client import RedisClient, RedisStoreConfig


@pytest.fixture(scope="session")
def redis_url() -> Iterator[str]:
    """Provide a Redis URL.

    In CI, REDIS_URL points at a service container. Locally, spin up a
    testcontainer if REDIS_URL is unset.
    """
    if url := os.getenv("REDIS_URL"):
        yield url
        return
    with RedisContainer("redis:7-alpine") as redis:
        host = redis.get_container_host_ip()
        port = redis.get_exposed_port(6379)
        yield f"redis://{host}:{port}"


@pytest.fixture
async def redis_client(redis_url: str):  # type: ignore[misc]
    cfg = RedisStoreConfig(url=redis_url, prefix="aktest")
    client = RedisClient(cfg)
    await client.redis.flushdb()  # type: ignore[no-untyped-call]
    yield client
    await client.redis.flushdb()  # type: ignore[no-untyped-call]
    await client.close()
