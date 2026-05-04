import asyncio
from datetime import UTC, datetime

import pytest

from agentkit._ids import EventId, MessageId, SessionId, TurnId, new_id
from agentkit.events import TextDelta
from agentkit.transports.redis_bus import RedisEventBus

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_publish_subscribe_round_trip(redis_client):  # type: ignore[no-untyped-def]
    bus = RedisEventBus(client=redis_client)
    sid = new_id(SessionId)

    received: asyncio.Queue[TextDelta] = asyncio.Queue()

    async def consume() -> None:
        async for ev in bus.subscribe(sid):
            await received.put(ev)  # type: ignore[arg-type]
            return  # drop after first

    consumer_task = asyncio.create_task(consume())

    # Brief delay to ensure subscriber is registered before publish.
    await asyncio.sleep(0.05)

    ev = TextDelta(
        event_id=new_id(EventId),
        session_id=sid,
        turn_id=new_id(TurnId),
        ts=datetime.now(UTC),
        sequence=1,
        message_id=new_id(MessageId),
        delta="hi",
    )
    await bus.publish(ev)

    got = await asyncio.wait_for(received.get(), timeout=2.0)
    assert isinstance(got, TextDelta)
    assert got.delta == "hi"

    consumer_task.cancel()
