import pytest

from agentkit._ids import CheckpointId, TurnId, new_id
from agentkit.store.checkpoint import approval_checkpoint_id
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


@pytest.mark.asyncio
async def test_list_ids_round_trips_real_redis_keys(redis_client):
    """SCAN gives back keys in the escaped layout; the ids must come back out
    usable as ``load`` arguments.

    The unit suite covers the same round-trip against a strict double
    (tests/unit/store/test_redis_checkpoint_enumeration.py) because this file
    needs a Docker daemon. This is the version that proves the real client's
    ``scan_iter`` and the real percent-encoding agree.
    """
    store = RedisCheckpointStore(redis_client)
    approvals = [approval_checkpoint_id(new_id(TurnId)) for _ in range(3)]
    for cid in approvals:
        await store.save(cid, b"{}")
    other = CheckpointId("some-other-subsystem:42")
    await store.save(other, b"{}")

    listed = await store.list_ids("approval:")

    assert sorted(listed) == sorted(approvals)
    for cid in listed:
        assert await store.load(cid) == b"{}"
    assert other not in listed
    assert sorted(await store.list_ids()) == sorted([*approvals, other])


@pytest.mark.asyncio
async def test_list_ids_stops_reporting_a_deleted_checkpoint(redis_client):
    store = RedisCheckpointStore(redis_client)
    cid = approval_checkpoint_id(new_id(TurnId))
    await store.save(cid, b"{}")
    assert await store.list_ids("approval:") == [cid]

    await store.delete(cid)

    assert await store.list_ids("approval:") == []
