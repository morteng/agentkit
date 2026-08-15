import asyncio
from datetime import UTC, datetime

import pytest

from agentkit._content import TextBlock
from agentkit._ids import MessageId, OwnerId, SessionId, new_id
from agentkit._messages import Message, MessageRole
from agentkit.store.redis.session import RedisSessionStore

pytestmark = pytest.mark.integration


def _msg(sid: SessionId, text: str) -> Message:
    return Message(
        id=new_id(MessageId),
        session_id=sid,
        role=MessageRole.USER,
        content=[TextBlock(text=text)],
        created_at=datetime.now(UTC),
    )


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


# ---- TTL coherence: a session must never outlive its own messages ----------


@pytest.mark.asyncio
async def test_touch_extends_the_messages_ttl_not_only_the_session(redis_client):
    """``touch`` used to refresh the session document alone.

    A session touched daily therefore kept its metadata alive forever while the
    messages list quietly ran out its original TTL — ``get()`` returned a
    session whose ``list_messages()`` was empty.
    """
    store = RedisSessionStore(redis_client, ttl_seconds=3600)
    sid = new_id(SessionId)
    await store.create(sid, OwnerId("u:1"))
    await store.append_message(sid, _msg(sid, "hello"))

    # Simulate a messages key most of the way through its life.
    await redis_client.redis.expire(redis_client.keys.messages(sid), 5)  # type: ignore[no-untyped-call]
    await store.touch(sid)

    msgs_ttl = await redis_client.redis.ttl(redis_client.keys.messages(sid))  # type: ignore[no-untyped-call]
    sess_ttl = await redis_client.redis.ttl(redis_client.keys.session(sid))  # type: ignore[no-untyped-call]
    assert msgs_ttl > 5, "touch left the messages list on its old, shorter TTL"
    assert 0 < msgs_ttl <= 3600
    assert 0 < sess_ttl <= 3600


@pytest.mark.asyncio
async def test_touch_on_a_session_with_no_messages_is_a_no_op(redis_client):
    """EXPIRE against a missing key returns 0 — not an error to propagate."""
    store = RedisSessionStore(redis_client, ttl_seconds=3600)
    sid = new_id(SessionId)
    await store.create(sid, OwnerId("u:1"))
    await store.touch(sid)
    assert await store.get(sid) is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("write", ["create", "append", "touch", "replace"])
async def test_owner_index_always_carries_a_ttl(redis_client, write):
    """The index used to be the one key in the layout that lived forever."""
    store = RedisSessionStore(redis_client, ttl_seconds=3600)
    owner = OwnerId("u:1")
    sid = new_id(SessionId)
    await store.create(sid, owner)
    if write == "append":
        await store.append_message(sid, _msg(sid, "hi"))
    elif write == "touch":
        await store.touch(sid)
    elif write == "replace":
        await store.replace(sid, [_msg(sid, "hi")])

    ttl = await redis_client.redis.ttl(redis_client.keys.owner_index(owner))  # type: ignore[no-untyped-call]
    assert 0 < ttl <= 3600, f"owner index has no expiry after {write}"


# ---- index hygiene ---------------------------------------------------------


@pytest.mark.asyncio
async def test_list_for_owner_prunes_ids_whose_session_is_gone(redis_client):
    """An expired session leaves its id in the ZSET. Read is where we learn the
    id is dead, so read is where it gets removed — otherwise the index is
    append-only for the life of the deployment."""
    store = RedisSessionStore(redis_client, ttl_seconds=3600)
    owner = OwnerId("u:1")
    alive = new_id(SessionId)
    await store.create(alive, owner)
    ghost = new_id(SessionId)
    await redis_client.redis.zadd(  # type: ignore[no-untyped-call]
        redis_client.keys.owner_index(owner), {str(ghost): 1.0}
    )

    summaries = await store.list_for_owner(owner)

    assert [s.id for s in summaries] == [alive]
    remaining = await redis_client.redis.zrange(  # type: ignore[no-untyped-call]
        redis_client.keys.owner_index(owner), 0, -1
    )
    decoded = {r.decode() if isinstance(r, bytes) else r for r in remaining}
    assert decoded == {str(alive)}, "dangling id survived a read"


@pytest.mark.asyncio
async def test_pruning_is_idempotent_and_leaves_a_healthy_index_alone(redis_client):
    store = RedisSessionStore(redis_client, ttl_seconds=3600)
    owner = OwnerId("u:1")
    s1 = new_id(SessionId)
    s2 = new_id(SessionId)
    await store.create(s1, owner)
    await store.create(s2, owner)
    assert len(await store.list_for_owner(owner)) == 2
    assert len(await store.list_for_owner(owner)) == 2


# ---- append atomicity ------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_appends_keep_the_count_and_the_list_in_agreement(redis_client):
    """The read-modify-write this replaced lost increments under concurrency:
    two appends both read N and both stored N+1, so the session permanently
    under-reported its own history. The count is now the list's LLEN, which
    cannot disagree with the list."""
    store = RedisSessionStore(redis_client, ttl_seconds=3600)
    sid = new_id(SessionId)
    await store.create(sid, OwnerId("u:1"))

    await asyncio.gather(*(store.append_message(sid, _msg(sid, f"m{i}")) for i in range(25)))

    sess = await store.get(sid)
    assert sess is not None
    stored = await store.list_messages(sid, limit=100)
    assert len(stored) == 25
    assert sess.message_count == 25


@pytest.mark.asyncio
async def test_message_count_reflects_the_list_even_if_the_document_is_stale(redis_client):
    """``message_count`` is derived on read. A document carrying a wrong count
    (an older writer, a partially applied migration) must not be believed."""
    store = RedisSessionStore(redis_client, ttl_seconds=3600)
    sid = new_id(SessionId)
    await store.create(sid, OwnerId("u:1"))
    await store.append_message(sid, _msg(sid, "one"))
    await store.append_message(sid, _msg(sid, "two"))

    # Corrupt the cached count directly, the way a lost increment used to.
    raw = await redis_client.redis.get(redis_client.keys.session(sid))  # type: ignore[no-untyped-call]
    tampered = raw.replace(b'"message_count":2', b'"message_count":0')
    assert tampered != raw
    await redis_client.redis.set(redis_client.keys.session(sid), tampered)  # type: ignore[no-untyped-call]

    sess = await store.get(sid)
    assert sess is not None
    assert sess.message_count == 2


@pytest.mark.asyncio
async def test_append_to_missing_session_still_raises_and_writes_nothing(redis_client):
    from agentkit.errors import StoreError

    store = RedisSessionStore(redis_client, ttl_seconds=3600)
    sid = new_id(SessionId)
    with pytest.raises(StoreError):
        await store.append_message(sid, _msg(sid, "orphan"))
    assert await store.list_messages(sid) == []
