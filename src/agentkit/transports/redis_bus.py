"""Redis pub/sub event fan-out for multi-replica deployments.

Orchestrators publish events to ``{prefix}:events:{session_id}``; bridges
subscribe and forward. Buffer (capped list) lets reconnects replay missed events.
"""

from collections.abc import AsyncIterator
from typing import cast

from agentkit._ids import SessionId
from agentkit.events import EVENT_ADAPTER, Event
from agentkit.events.base import BaseEvent
from agentkit.store.redis.client import RedisClient


class RedisEventBus:
    def __init__(self, *, client: RedisClient, buffer_max_events: int = 200) -> None:
        self._c = client
        self._buf_max = buffer_max_events

    async def publish(self, event: BaseEvent) -> None:
        channel = self._c.keys.event_channel(event.session_id)
        buffer_key = self._c.keys.event_buffer(event.session_id)
        payload = event.model_dump_json()
        async with self._c.redis.pipeline(transaction=False) as pipe:  # type: ignore[no-untyped-call]
            pipe.publish(channel, payload)  # type: ignore[no-untyped-call]
            pipe.rpush(buffer_key, payload)  # type: ignore[no-untyped-call]
            pipe.ltrim(buffer_key, -self._buf_max, -1)  # type: ignore[no-untyped-call]
            await pipe.execute()  # type: ignore[no-untyped-call]

    async def subscribe(self, session_id: SessionId) -> AsyncIterator[Event]:
        channel = self._c.keys.event_channel(session_id)
        pubsub = self._c.redis.pubsub()  # type: ignore[no-untyped-call]
        await pubsub.subscribe(channel)  # type: ignore[no-untyped-call]
        try:
            async for message in pubsub.listen():  # type: ignore[no-untyped-call,reportUnknownVariableType]
                msg = cast("dict[str, object]", message)
                if msg.get("type") != "message":
                    continue
                data = msg["data"]
                if isinstance(data, bytes):
                    data = data.decode("utf-8")
                yield EVENT_ADAPTER.validate_json(cast("str | bytes", data))
        finally:
            await pubsub.unsubscribe(channel)  # type: ignore[no-untyped-call]
            await pubsub.aclose()  # type: ignore[no-untyped-call]

    async def replay_buffer(self, session_id: SessionId, *, since_sequence: int = 0) -> list[Event]:
        buffer_key = self._c.keys.event_buffer(session_id)
        raws: list[bytes] = await self._c.redis.lrange(  # type: ignore[no-untyped-call,reportUnknownVariableType]
            buffer_key, 0, -1
        )
        events: list[Event] = []
        for raw in raws:  # type: ignore[reportUnknownVariableType]
            ev = EVENT_ADAPTER.validate_json(cast("bytes", raw))
            if ev.sequence > since_sequence:
                events.append(ev)
        return events
