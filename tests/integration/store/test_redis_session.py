from datetime import UTC, datetime

import pytest

from agentkit._content import TextBlock
from agentkit._ids import MessageId, OwnerId, SessionId, new_id
from agentkit._messages import Message, MessageRole
from agentkit.store.redis.session import RedisSessionStore

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_create_get_round_trip(redis_client):
    store = RedisSessionStore(redis_client)
    sid = new_id(SessionId)
    sess = await store.create(sid, OwnerId("u:1"), title="hi")
    assert (await store.get(sid)) == sess


@pytest.mark.asyncio
async def test_append_and_list_messages(redis_client):
    store = RedisSessionStore(redis_client)
    sid = new_id(SessionId)
    await store.create(sid, OwnerId("u:1"))
    msg = Message(
        id=new_id(MessageId),
        session_id=sid,
        role=MessageRole.USER,
        content=[TextBlock(text="hello")],
        created_at=datetime.now(UTC),
    )
    await store.append_message(sid, msg)
    msgs = await store.list_messages(sid)
    assert len(msgs) == 1
    assert isinstance(msgs[0].content[0], TextBlock)
    assert msgs[0].content[0].text == "hello"


@pytest.mark.asyncio
async def test_list_for_owner_orders_by_recency(redis_client):
    store = RedisSessionStore(redis_client)
    owner = OwnerId("u:1")
    s1 = new_id(SessionId)
    s2 = new_id(SessionId)
    await store.create(s1, owner)
    await store.create(s2, owner)
    await store.touch(s1)  # s1 becomes most recent
    summaries = await store.list_for_owner(owner)
    assert [s.id for s in summaries] == [s1, s2]


@pytest.mark.asyncio
async def test_delete_removes_session_and_messages(redis_client):
    store = RedisSessionStore(redis_client)
    sid = new_id(SessionId)
    await store.create(sid, OwnerId("u:1"))
    await store.delete(sid)
    assert await store.get(sid) is None
    assert await store.list_messages(sid) == []
