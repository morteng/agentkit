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


@pytest.mark.asyncio
async def test_replace_swaps_messages_atomically(redis_client):
    store = RedisSessionStore(redis_client)
    sid = new_id(SessionId)
    await store.create(sid, OwnerId("u:1"))
    for text in ("one", "two", "three"):
        await store.append_message(
            sid,
            Message(
                id=new_id(MessageId),
                session_id=sid,
                role=MessageRole.USER,
                content=[TextBlock(text=text)],
                created_at=datetime.now(UTC),
            ),
        )
    sess = await store.get(sid)
    assert sess is not None
    assert sess.message_count == 3

    replacement = [
        Message(
            id=new_id(MessageId),
            session_id=sid,
            role=MessageRole.USER,
            content=[TextBlock(text="summary")],
            created_at=datetime.now(UTC),
        )
    ]
    await store.replace(sid, replacement)

    msgs = await store.list_messages(sid)
    assert len(msgs) == 1
    assert isinstance(msgs[0].content[0], TextBlock)
    assert msgs[0].content[0].text == "summary"
    sess = await store.get(sid)
    assert sess is not None
    assert sess.message_count == 1


@pytest.mark.asyncio
async def test_replace_preserves_ttl_on_messages_key(redis_client):
    """Redis TTL survives the DEL+RPUSH swap inside the MULTI/EXEC pipeline."""
    store = RedisSessionStore(redis_client, ttl_seconds=3600)
    sid = new_id(SessionId)
    await store.create(sid, OwnerId("u:1"))
    await store.append_message(
        sid,
        Message(
            id=new_id(MessageId),
            session_id=sid,
            role=MessageRole.USER,
            content=[TextBlock(text="one")],
            created_at=datetime.now(UTC),
        ),
    )
    await store.replace(
        sid,
        [
            Message(
                id=new_id(MessageId),
                session_id=sid,
                role=MessageRole.USER,
                content=[TextBlock(text="summary")],
                created_at=datetime.now(UTC),
            )
        ],
    )
    ttl = await redis_client.redis.ttl(redis_client.keys.messages(sid))  # type: ignore[no-untyped-call]
    assert 0 < ttl <= 3600
