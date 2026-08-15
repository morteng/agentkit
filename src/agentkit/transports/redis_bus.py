"""Redis pub/sub event fan-out for multi-replica deployments.

Orchestrators publish events to ``{prefix}:events:{session_id}``; bridges
subscribe and forward. Buffer (capped list) lets reconnects replay missed events.

Replay cursor
-------------
``BaseEvent.sequence`` is **per turn** and restarts at 0 on every turn, so it
is not a session-wide cursor: filtering the buffer on ``sequence`` alone drops
the whole current turn as soon as its numbering falls back under the last
number seen in an earlier turn, and drops every turn's ``sequence == 0`` event
unconditionally. The cursor is therefore the pair ``(turn_id, sequence)`` —
``turn_id`` is already on every event — and replay resumes at the first event
after that position in buffer order.
"""

from collections.abc import AsyncIterator
from typing import cast

from agentkit._ids import SessionId, TurnId
from agentkit._logging import get_logger
from agentkit.events import EVENT_ADAPTER, Event
from agentkit.events.base import BaseEvent
from agentkit.store.redis.client import RedisClient

log = get_logger(__name__)

#: Default lifetime of a session's replay buffer. Matches
#: :class:`agentkit.store.redis.session.RedisSessionStore`'s own default TTL so
#: the buffer expires alongside the session it belongs to instead of pinning
#: the events of every session the deployment has ever run.
DEFAULT_EVENT_BUFFER_TTL_SECONDS = 30 * 24 * 60 * 60


class RedisEventBus:
    """Publish/subscribe fan-out plus a bounded, expiring per-session replay buffer.

    ``buffer_max_events`` caps the buffer length; ``buffer_ttl_seconds`` caps
    its lifetime and is refreshed on every publish (a sliding window, matching
    how the session store treats session keys).
    """

    def __init__(
        self,
        *,
        client: RedisClient,
        buffer_max_events: int = 200,
        buffer_ttl_seconds: int = DEFAULT_EVENT_BUFFER_TTL_SECONDS,
    ) -> None:
        self._c = client
        self._buf_max = buffer_max_events
        self._buf_ttl = buffer_ttl_seconds

    async def publish(self, event: BaseEvent) -> None:
        channel = self._c.keys.event_channel(event.session_id)
        buffer_key = self._c.keys.event_buffer(event.session_id)
        payload = event.model_dump_json()
        async with self._c.redis.pipeline(transaction=False) as pipe:  # type: ignore[no-untyped-call]
            pipe.publish(channel, payload)  # type: ignore[no-untyped-call]
            pipe.rpush(buffer_key, payload)  # type: ignore[no-untyped-call]
            pipe.ltrim(buffer_key, -self._buf_max, -1)  # type: ignore[no-untyped-call]
            # Without this the buffer is immortal: ltrim bounds its length but
            # nothing bounds its lifetime, so every session that ever ran keeps
            # its last N events in Redis forever.
            pipe.expire(buffer_key, self._buf_ttl)  # type: ignore[no-untyped-call]
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

    async def replay_buffer(
        self,
        session_id: SessionId,
        *,
        since_turn_id: TurnId | None = None,
        since_sequence: int = 0,
    ) -> list[Event]:
        """Buffered events strictly after the cursor ``(since_turn_id, since_sequence)``.

        Pass the ``turn_id`` and ``sequence`` of the last event the client
        actually received. Replay then returns the rest of that turn plus
        every later turn, in publish order:

        * events belonging to ``since_turn_id`` with ``sequence >
          since_sequence``, and
        * every event of every turn that appears after ``since_turn_id`` in the
          buffer.

        Earlier turns are considered delivered and are skipped. If
        ``since_turn_id`` is not in the buffer at all — the client was gone
        longer than ``buffer_max_events`` — the whole buffer is returned and a
        warning is logged: over-delivering a duplicate is recoverable, silently
        skipping a gap is not.

        With no ``since_turn_id`` the whole buffer is returned; ``sequence`` on
        its own cannot identify a position in a multi-turn stream. Passing
        ``since_sequence`` without ``since_turn_id`` keeps the pre-existing
        single-turn filter for backward compatibility and logs that it is
        ambiguous.
        """
        buffer_key = self._c.keys.event_buffer(session_id)
        raws: list[bytes] = await self._c.redis.lrange(  # type: ignore[no-untyped-call,reportUnknownVariableType]
            buffer_key, 0, -1
        )
        buffered: list[Event] = [
            EVENT_ADAPTER.validate_json(cast("bytes", raw))  # type: ignore[reportUnknownArgumentType]
            for raw in raws  # type: ignore[reportUnknownVariableType]
        ]

        if since_turn_id is None:
            if since_sequence <= 0:
                return buffered
            log.warning(
                "event_replay_sequence_only_cursor",
                session_id=str(session_id),
                since_sequence=since_sequence,
                hint="sequence restarts per turn; pass since_turn_id for a correct cursor",
            )
            return [ev for ev in buffered if ev.sequence > since_sequence]

        events: list[Event] = []
        seen_cursor_turn = False
        for ev in buffered:
            if ev.turn_id == since_turn_id:
                seen_cursor_turn = True
                if ev.sequence > since_sequence:
                    events.append(ev)
            elif seen_cursor_turn:
                events.append(ev)
        if not seen_cursor_turn and buffered:
            log.warning(
                "event_replay_cursor_evicted",
                session_id=str(session_id),
                since_turn_id=str(since_turn_id),
                buffered=len(buffered),
                hint="cursor turn is no longer in the buffer; replaying everything held",
            )
            return buffered
        return events
