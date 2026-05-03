from datetime import UTC, datetime

import pytest

from agentkit._content import TextBlock
from agentkit._ids import CheckpointId, MessageId, OwnerId, SessionId, new_id
from agentkit._messages import Message, MessageRole
from agentkit.store import MemoryScope, MemoryValue
from agentkit.store.fakes import FakeCheckpointStore, FakeMemoryStore, FakeSessionStore


@pytest.mark.asyncio
async def test_session_store_create_and_append_and_list():
    store = FakeSessionStore()
    sid = new_id(SessionId)
    owner = OwnerId("user:abc")
    sess = await store.create(sid, owner, title="hello")
    assert sess.id == sid

    msg = Message(
        id=new_id(MessageId),
        session_id=sid,
        role=MessageRole.USER,
        content=[TextBlock(text="hi")],
        created_at=datetime.now(UTC),
    )
    await store.append_message(sid, msg)
    msgs = await store.list_messages(sid)
    assert len(msgs) == 1
    assert msgs[0].id == msg.id

    summaries = await store.list_for_owner(owner)
    assert len(summaries) == 1
    assert summaries[0].id == sid


@pytest.mark.asyncio
async def test_session_store_get_returns_none_for_missing():
    store = FakeSessionStore()
    assert await store.get(new_id(SessionId)) is None


@pytest.mark.asyncio
async def test_memory_store_save_recall_search():
    store = FakeMemoryStore()
    scope = MemoryScope(namespace="t", user_id="u1")
    now = datetime.now(UTC)
    await store.save(
        scope, "k1", MemoryValue(text="user prefers SI units", created_at=now, updated_at=now)
    )
    await store.save(
        scope, "k2", MemoryValue(text="user lives in Oslo", created_at=now, updated_at=now)
    )

    v = await store.recall(scope, "k1")
    assert v is not None and "SI units" in v.text

    hits = await store.search(scope, "Oslo")
    assert len(hits) == 1
    assert hits[0].key == "k2"


@pytest.mark.asyncio
async def test_memory_store_isolates_scopes():
    store = FakeMemoryStore()
    a = MemoryScope(namespace="t", user_id="u1")
    b = MemoryScope(namespace="t", user_id="u2")
    now = datetime.now(UTC)
    val = MemoryValue(text="A", created_at=now, updated_at=now)
    await store.save(a, "k", val)
    assert await store.recall(b, "k") is None


@pytest.mark.asyncio
async def test_checkpoint_store_round_trips_bytes():
    store = FakeCheckpointStore()
    cid = new_id(CheckpointId)
    payload = b"\x00\x01\x02"
    await store.save(cid, payload)
    assert await store.load(cid) == payload
    await store.delete(cid)
    assert await store.load(cid) is None
