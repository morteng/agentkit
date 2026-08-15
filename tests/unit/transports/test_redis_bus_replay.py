"""Replay-cursor and TTL semantics for :class:`RedisEventBus`.

Two audit findings live here, both about the buffer rather than the pub/sub
path, so they are exercised against an in-memory stand-in for Redis instead of
a container:

* ``replay_buffer`` filtered on ``sequence`` alone, but ``sequence`` restarts
  at 0 every turn. A client reconnecting mid-conversation was handed nothing
  from the turn it was actually in.
* the buffer key had no expiry, so every session the deployment ever ran kept
  its last N events in Redis forever.
"""

from collections.abc import Sequence
from datetime import UTC, datetime

from agentkit._ids import EventId, MessageId, SessionId, TurnId, new_id
from agentkit.events import Event, TextDelta
from agentkit.store.redis.keys import KeyBuilder
from agentkit.transports.redis_bus import DEFAULT_EVENT_BUFFER_TTL_SECONDS, RedisEventBus


class _FakePipeline:
    def __init__(self, redis: "_FakeRedis") -> None:
        self._r = redis
        self._ops: list[tuple[str, tuple[object, ...]]] = []

    async def __aenter__(self) -> "_FakePipeline":
        return self

    async def __aexit__(self, *_exc: object) -> bool:
        return False

    def publish(self, channel: str, payload: str) -> None:
        self._ops.append(("publish", (channel, payload)))

    def rpush(self, key: str, payload: str) -> None:
        self._ops.append(("rpush", (key, payload)))

    def ltrim(self, key: str, start: int, end: int) -> None:
        self._ops.append(("ltrim", (key, start, end)))

    def expire(self, key: str, ttl: int) -> None:
        self._ops.append(("expire", (key, ttl)))

    async def execute(self) -> None:
        for op, args in self._ops:
            getattr(self._r, f"_apply_{op}")(*args)
        self._ops.clear()


class _FakeRedis:
    def __init__(self) -> None:
        self.lists: dict[str, list[bytes]] = {}
        self.published: list[tuple[str, str]] = []
        self.ttls: dict[str, int] = {}

    def pipeline(self, transaction: bool = True) -> _FakePipeline:
        _ = transaction
        return _FakePipeline(self)

    async def lrange(self, key: str, start: int, end: int) -> list[bytes]:
        values = self.lists.get(key, [])
        return values[start:] if end == -1 else values[start : end + 1]

    # ---- pipeline command application --------------------------------------

    def _apply_publish(self, channel: str, payload: str) -> None:
        self.published.append((channel, payload))

    def _apply_rpush(self, key: str, payload: str) -> None:
        self.lists.setdefault(key, []).append(payload.encode("utf-8"))

    def _apply_ltrim(self, key: str, start: int, end: int) -> None:
        values = self.lists.get(key, [])
        self.lists[key] = values[start:] if end == -1 else values[start : end + 1]

    def _apply_expire(self, key: str, ttl: int) -> None:
        self.ttls[key] = ttl


class _FakeRedisClient:
    def __init__(self) -> None:
        self.redis = _FakeRedis()
        self.keys = KeyBuilder(prefix="aktest")


def _bus(**kwargs: object) -> tuple[RedisEventBus, _FakeRedisClient]:
    client = _FakeRedisClient()
    bus = RedisEventBus(client=client, **kwargs)  # type: ignore[arg-type]
    return bus, client


def _delta(session_id: SessionId, turn_id: TurnId, sequence: int, text: str) -> TextDelta:
    return TextDelta(
        event_id=new_id(EventId),
        session_id=session_id,
        turn_id=turn_id,
        ts=datetime.now(UTC),
        sequence=sequence,
        message_id=new_id(MessageId),
        delta=text,
        block_index=0,
    )


def _texts(events: Sequence[Event]) -> list[str]:
    """The ``delta`` of each replayed event, asserting they really are deltas.

    ``replay_buffer`` returns events typed as the ``Event`` union, and ``delta``
    exists only on ``TextDelta``. Reading it straight off the union is what
    pyright objects to, and the objection is worth honouring rather than
    silencing: if replay ever handed back the wrong event type, a bare
    ``ev.delta`` would fail with an ``AttributeError`` inside a list
    comprehension — which reads as a broken test, not a broken bus. Narrowing
    here turns that into a named assertion instead.
    """
    out: list[str] = []
    for ev in events:
        assert isinstance(ev, TextDelta), f"expected TextDelta, got {type(ev).__name__}"
        out.append(ev.delta)
    return out


# ---- TTL --------------------------------------------------------------------


async def test_publish_expires_the_buffer_key():
    bus, client = _bus()
    sid = new_id(SessionId)
    await bus.publish(_delta(sid, new_id(TurnId), 0, "hi"))

    key = client.keys.event_buffer(sid)
    assert client.redis.ttls[key] == DEFAULT_EVENT_BUFFER_TTL_SECONDS


async def test_buffer_ttl_is_configurable_and_refreshed_on_every_publish():
    bus, client = _bus(buffer_ttl_seconds=60)
    sid = new_id(SessionId)
    turn = new_id(TurnId)
    key = client.keys.event_buffer(sid)

    await bus.publish(_delta(sid, turn, 0, "a"))
    client.redis.ttls[key] = 3  # simulate the key ageing towards expiry
    await bus.publish(_delta(sid, turn, 1, "b"))

    assert client.redis.ttls[key] == 60


async def test_buffer_is_still_length_capped():
    bus, _client = _bus(buffer_max_events=3)
    sid = new_id(SessionId)
    turn = new_id(TurnId)
    for i in range(6):
        await bus.publish(_delta(sid, turn, i, f"e{i}"))

    replayed = await bus.replay_buffer(sid)
    assert [ev.sequence for ev in replayed] == [3, 4, 5]


# ---- cursor -----------------------------------------------------------------


async def test_replay_without_a_cursor_returns_everything_including_sequence_zero():
    """The old sequence-only filter dropped every turn's first event."""
    bus, _ = _bus()
    sid = new_id(SessionId)
    turn = new_id(TurnId)
    for i in range(3):
        await bus.publish(_delta(sid, turn, i, f"e{i}"))

    replayed = await bus.replay_buffer(sid)
    assert [ev.sequence for ev in replayed] == [0, 1, 2]


async def test_replay_resumes_inside_the_cursor_turn():
    bus, _ = _bus()
    sid = new_id(SessionId)
    turn = new_id(TurnId)
    for i in range(5):
        await bus.publish(_delta(sid, turn, i, f"e{i}"))

    replayed = await bus.replay_buffer(sid, since_turn_id=turn, since_sequence=2)
    assert [ev.sequence for ev in replayed] == [3, 4]


async def test_replay_does_not_drop_a_new_turn_whose_sequence_restarted():
    """The headline bug: sequence resets to 0, so a sequence-only cursor eats the
    entire current turn.

    The client last saw ``(turn_a, 4)``. Everything in ``turn_b`` is new, even
    though its sequence numbers are all lower.
    """
    bus, _ = _bus()
    sid = new_id(SessionId)
    turn_a, turn_b = new_id(TurnId), new_id(TurnId)
    for i in range(5):
        await bus.publish(_delta(sid, turn_a, i, f"a{i}"))
    for i in range(3):
        await bus.publish(_delta(sid, turn_b, i, f"b{i}"))

    replayed = await bus.replay_buffer(sid, since_turn_id=turn_a, since_sequence=4)
    assert _texts(replayed) == ["b0", "b1", "b2"]


async def test_replay_returns_the_rest_of_the_cursor_turn_and_all_later_turns():
    bus, _ = _bus()
    sid = new_id(SessionId)
    turn_a, turn_b = new_id(TurnId), new_id(TurnId)
    for i in range(4):
        await bus.publish(_delta(sid, turn_a, i, f"a{i}"))
    for i in range(2):
        await bus.publish(_delta(sid, turn_b, i, f"b{i}"))

    replayed = await bus.replay_buffer(sid, since_turn_id=turn_a, since_sequence=1)
    assert _texts(replayed) == ["a2", "a3", "b0", "b1"]


async def test_replay_skips_turns_that_completed_before_the_cursor():
    bus, _ = _bus()
    sid = new_id(SessionId)
    old, current = new_id(TurnId), new_id(TurnId)
    for i in range(3):
        await bus.publish(_delta(sid, old, i, f"old{i}"))
    for i in range(3):
        await bus.publish(_delta(sid, current, i, f"cur{i}"))

    replayed = await bus.replay_buffer(sid, since_turn_id=current, since_sequence=0)
    assert _texts(replayed) == ["cur1", "cur2"]


async def test_replay_returns_the_whole_buffer_when_the_cursor_turn_was_evicted():
    """Over-delivering a duplicate is recoverable; a silent gap is not."""
    bus, _ = _bus(buffer_max_events=2)
    sid = new_id(SessionId)
    evicted, current = new_id(TurnId), new_id(TurnId)
    await bus.publish(_delta(sid, evicted, 0, "gone"))
    for i in range(2):
        await bus.publish(_delta(sid, current, i, f"cur{i}"))

    replayed = await bus.replay_buffer(sid, since_turn_id=evicted, since_sequence=0)
    assert _texts(replayed) == ["cur0", "cur1"]


async def test_replay_of_an_empty_buffer_is_empty_even_with_a_cursor():
    bus, _ = _bus()
    sid = new_id(SessionId)
    assert await bus.replay_buffer(sid, since_turn_id=new_id(TurnId), since_sequence=7) == []


async def test_sequence_only_cursor_keeps_its_legacy_behaviour():
    """Back-compat: callers passing only ``since_sequence`` still get the old filter."""
    bus, _ = _bus()
    sid = new_id(SessionId)
    turn = new_id(TurnId)
    for i in range(4):
        await bus.publish(_delta(sid, turn, i, f"e{i}"))

    replayed = await bus.replay_buffer(sid, since_sequence=1)
    assert [ev.sequence for ev in replayed] == [2, 3]
