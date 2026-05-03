import pytest

from agentkit._ids import CheckpointId, new_id
from agentkit.store.redis.checkpoint import RedisCheckpointStore

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_save_load_delete(redis_client):
    store = RedisCheckpointStore(redis_client)
    cid = new_id(CheckpointId)
    await store.save(cid, b"payload")
    assert await store.load(cid) == b"payload"
    await store.delete(cid)
    assert await store.load(cid) is None
