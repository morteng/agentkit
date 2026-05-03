from datetime import UTC, datetime

import pytest

from agentkit.store.memory import MemoryScope, MemoryValue
from agentkit.store.redis.memory import RedisMemoryStore

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_save_recall_round_trip(redis_client):
    store = RedisMemoryStore(redis_client)
    scope = MemoryScope(namespace="t", user_id="u1")
    now = datetime.now(UTC)
    await store.save(scope, "k", MemoryValue(text="hi", created_at=now, updated_at=now))
    v = await store.recall(scope, "k")
    assert v is not None and v.text == "hi"


@pytest.mark.asyncio
async def test_search_finds_substring(redis_client):
    store = RedisMemoryStore(redis_client)
    scope = MemoryScope(namespace="t", user_id="u1")
    now = datetime.now(UTC)
    await store.save(
        scope, "k1", MemoryValue(text="user lives in Oslo", created_at=now, updated_at=now)
    )
    await store.save(
        scope, "k2", MemoryValue(text="user prefers SI units", created_at=now, updated_at=now)
    )
    hits = await store.search(scope, "Oslo")
    assert {h.key for h in hits} == {"k1"}
