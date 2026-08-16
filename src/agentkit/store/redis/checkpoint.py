"""Redis-backed CheckpointStore. Raw-bytes payloads with TTL."""

from typing import TYPE_CHECKING, Any, cast
from urllib.parse import unquote

from agentkit._ids import CheckpointId
from agentkit.store.checkpoint import CheckpointPayload, CheckpointStore
from agentkit.store.redis.client import RedisClient

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

_SCAN_BATCH = 500
"""``count`` hint per SCAN round-trip. Not a limit on the result — SCAN's
``count`` only sizes the work each cursor step does."""


class RedisCheckpointStore(CheckpointStore):
    def __init__(self, client: RedisClient, *, ttl_seconds: int = 24 * 60 * 60) -> None:
        self._c = client
        self._ttl = ttl_seconds

    async def save(self, checkpoint_id: CheckpointId, payload: CheckpointPayload) -> None:
        await self._c.redis.set(  # type: ignore[no-untyped-call]
            self._c.keys.checkpoint(checkpoint_id),
            payload,
            ex=self._ttl,
        )

    async def load(self, checkpoint_id: CheckpointId) -> CheckpointPayload | None:
        return await self._c.redis.get(self._c.keys.checkpoint(checkpoint_id))  # type: ignore[no-untyped-call]

    async def delete(self, checkpoint_id: CheckpointId) -> None:
        await self._c.redis.delete(self._c.keys.checkpoint(checkpoint_id))  # type: ignore[no-untyped-call]

    async def list_ids(self, prefix: str = "") -> list[CheckpointId]:
        """Enumerate stored checkpoint ids — see :class:`EnumerableCheckpointStore`.

        SCAN rather than a maintained index SET, which is the opposite of the
        choice :class:`~agentkit.store.redis.memory.RedisMemoryStore` makes for
        ``list_keys``, for two reasons specific to checkpoints:

        * These entries carry a TTL and that store's do not. An index SET would
          keep naming checkpoints Redis had already dropped, so every reader
          would need to prune on miss — a second write path, on the read side,
          racing every other reader. SCAN sees exactly what is still there.
        * ``list_keys`` is called from ``search``, once per query. This is
          called by a periodic expiry sweep. The O(keyspace) cost that rules
          SCAN out of a hot path is unremarkable at sweep frequency, and
          ``scan_iter`` yields to the event loop between batches rather than
          blocking the server the way ``KEYS`` would.

        The ``prefix`` is matched against the *unescaped* checkpoint id, so it
        has to be escaped before it goes into the glob. That is also what makes
        the glob injection-safe without any quoting of its own: percent-
        encoding leaves only the RFC 3986 unreserved set, and ``*``, ``?``,
        ``[`` and ``\\`` are all outside it, so no prefix can produce a pattern
        that matches more than it names.
        """
        pattern = f"{self._c.keys.checkpoint(CheckpointId(prefix))}*"
        # Where the escaped prefix ends and the escaped id continues. Slicing
        # by this length is why the pattern above is built from the same
        # KeyBuilder call the keys themselves are: any change to the key layout
        # moves both together.
        key_prefix_len = len(self._c.keys.checkpoint(CheckpointId("")))
        ids: list[CheckpointId] = []
        # ``decode_responses`` is a deployment setting, so keys arrive as bytes
        # or str depending on how the pool was built — same defensive decode
        # RedisMemoryStore.list_keys and RedisSessionStore do.
        scanned = cast(
            "AsyncIterator[Any]",
            self._c.redis.scan_iter(match=pattern, count=_SCAN_BATCH),  # pyright: ignore[reportUnknownMemberType]
        )
        async for raw in scanned:
            key: str = raw.decode() if isinstance(raw, bytes) else str(raw)
            ids.append(CheckpointId(unquote(key[key_prefix_len:])))
        return ids
