"""Redis-backed SessionStore.

Layout:
- ``{prefix}:sess:{id}``               JSON session metadata
- ``{prefix}:msgs:{id}``               LIST of message JSONs (RPUSH on append)
- ``{prefix}:owner:{owner}:sessions``  ZSET (score = updated_at unix ms) of session ids

TTL: every key an owner's data lives in — the session doc, its messages list,
*and* the owner index — expires after ``ttl_seconds`` of inactivity (default 30
days), and every write path refreshes all three together. Refreshing only some
of them is what produced the two bugs this layout used to have: ``touch``
extended the session but not its messages, so an active session eventually
answered ``get()`` with metadata and ``list_messages()`` with an empty list;
and the owner index had no expiry at all, so it grew a permanent tombstone for
every session that ever existed. ``list_for_owner`` additionally prunes ids
whose session is already gone, so an index that predates this fix drains itself
as it is read.

``message_count`` is DERIVED, not accumulated: ``get`` reads it from the
messages list's own LLEN rather than trusting the number cached in the session
document. The cached number was maintained by a read-modify-write across four
unsynchronised round trips, so two concurrent appends both read N and both
wrote N+1 and the count stayed permanently short. A value the list itself
already knows should never have been re-derived by arithmetic. The document
still carries a ``message_count`` for anyone eyeballing Redis directly; it is
advisory, and the list wins.
"""

from datetime import UTC, datetime
from typing import Any, cast

from agentkit._ids import OwnerId, SessionId
from agentkit._logging import get_logger
from agentkit._messages import Message
from agentkit.errors import StoreError
from agentkit.store.redis.client import RedisClient
from agentkit.store.redis.serialization import from_versioned_json, to_versioned_json
from agentkit.store.session import Session, SessionStore, SessionSummary

log = get_logger(__name__)

_SCHEMA_V = 1


class RedisSessionStore(SessionStore):
    def __init__(self, client: RedisClient, *, ttl_seconds: int = 30 * 24 * 60 * 60) -> None:
        self._c = client
        self._ttl = ttl_seconds

    async def create(
        self,
        session_id: SessionId,
        owner: OwnerId,
        metadata: dict[str, str] | None = None,
        title: str | None = None,
    ) -> Session:
        now = datetime.now(UTC)
        sess = Session(
            id=session_id,
            owner=owner,
            title=title,
            metadata=metadata or {},
            created_at=now,
            updated_at=now,
            message_count=0,
        )
        await self._save_session(sess)
        await self._index_owner(owner, session_id, now)
        return sess

    async def get(self, session_id: SessionId) -> Session | None:
        """Load a session, taking ``message_count`` from the messages list.

        One round trip: the document and the list length are pipelined
        together, so reading the truthful count costs nothing extra in latency.
        """
        async with self._c.redis.pipeline(transaction=False) as pipe:  # type: ignore[no-untyped-call]
            pipe.get(self._c.keys.session(session_id))  # type: ignore[no-untyped-call]
            pipe.llen(self._c.keys.messages(session_id))  # type: ignore[no-untyped-call]
            raw, stored_count = cast("tuple[Any, Any]", tuple(await pipe.execute()))  # type: ignore[no-untyped-call]
        if raw is None:
            return None
        assert isinstance(raw, bytes)
        return self._decode_session(raw, message_count=int(stored_count))

    async def append_message(self, session_id: SessionId, message: Message) -> None:
        """Append one message; push, document and index land in one MULTI/EXEC.

        This used to be four unsynchronised round trips (GET, RPUSH, EXPIRE,
        SET, ZADD) wrapped around a read-modify-write of ``message_count``. Two
        of those problems are gone rather than mitigated:

        * The count is no longer accumulated here at all — ``get`` derives it
          from LLEN — so concurrent appends cannot lose increments. The number
          written into the document is advisory bookkeeping.
        * Everything that mutates state is one transaction, so a crash or a
          dropped connection can no longer leave a message in the list that the
          session document and the owner index have never heard of.

        What remains is a read (the document, to carry ``owner``/``title``/
        ``metadata`` forward) followed by the transaction. Two appends racing
        there both write ``updated_at`` and the later write wins, which is the
        correct answer for a timestamp; nothing else in the document is
        derived from what was read.
        """
        sess = await self.get(session_id)
        if sess is None:
            raise StoreError(f"session not found: {session_id}")
        updated = sess.model_copy(
            update={
                "message_count": sess.message_count + 1,
                "updated_at": datetime.now(UTC),
            }
        )
        msgs_key = self._c.keys.messages(session_id)
        owner_key = self._c.keys.owner_index(updated.owner)
        encoded = to_versioned_json(message.model_dump(mode="json"), schema_version=_SCHEMA_V)
        async with self._c.redis.pipeline(transaction=True) as pipe:  # type: ignore[no-untyped-call]
            pipe.rpush(msgs_key, encoded)  # type: ignore[no-untyped-call]
            pipe.expire(msgs_key, self._ttl)  # type: ignore[no-untyped-call]
            pipe.set(  # type: ignore[no-untyped-call]
                self._c.keys.session(session_id),
                to_versioned_json(updated.model_dump(mode="json"), schema_version=_SCHEMA_V),
                ex=self._ttl,
            )
            pipe.zadd(  # type: ignore[no-untyped-call]
                owner_key,
                {str(session_id): updated.updated_at.timestamp() * 1000},
            )
            pipe.expire(owner_key, self._ttl)  # type: ignore[no-untyped-call]
            await pipe.execute()  # type: ignore[no-untyped-call]

    async def replace(self, session_id: SessionId, messages: list[Message]) -> None:
        sess = await self.get(session_id)
        if sess is None:
            raise StoreError(f"session not found: {session_id}")
        updated = sess.model_copy(
            update={
                "message_count": len(messages),
                "updated_at": datetime.now(UTC),
            }
        )
        msgs_key = self._c.keys.messages(session_id)
        sess_key = self._c.keys.session(session_id)
        encoded = [
            to_versioned_json(m.model_dump(mode="json"), schema_version=_SCHEMA_V) for m in messages
        ]
        # MULTI/EXEC: delete + repopulate + refresh session doc as one atomic
        # swap, so a concurrent reader never observes the messages list
        # mid-replace (empty after DEL but before the new RPUSH lands).
        async with self._c.redis.pipeline(transaction=True) as pipe:  # type: ignore[no-untyped-call]
            pipe.delete(msgs_key)  # type: ignore[no-untyped-call]
            if encoded:
                pipe.rpush(msgs_key, *encoded)  # type: ignore[no-untyped-call]
                pipe.expire(msgs_key, self._ttl)  # type: ignore[no-untyped-call]
            pipe.set(  # type: ignore[no-untyped-call]
                sess_key,
                to_versioned_json(updated.model_dump(mode="json"), schema_version=_SCHEMA_V),
                ex=self._ttl,
            )
            owner_key = self._c.keys.owner_index(updated.owner)
            pipe.zadd(  # type: ignore[no-untyped-call]
                owner_key,
                {str(session_id): updated.updated_at.timestamp() * 1000},
            )
            pipe.expire(owner_key, self._ttl)  # type: ignore[no-untyped-call]
            await pipe.execute()  # type: ignore[no-untyped-call]

    async def list_messages(self, session_id: SessionId, *, limit: int = 200) -> list[Message]:
        raws: list[bytes] = await self._c.redis.lrange(  # type: ignore[no-untyped-call,reportUnknownVariableType]
            self._c.keys.messages(session_id), -limit, -1
        )
        return [
            Message.model_validate(from_versioned_json(cast("bytes", r))[0])
            for r in raws  # type: ignore[reportUnknownVariableType]
        ]

    async def list_for_owner(self, owner: OwnerId, *, limit: int = 30) -> list[SessionSummary]:
        """List an owner's most recent sessions, pruning dead index entries.

        The index is a ZSET of ids, not of documents, so a session that expired
        (or was deleted by a path that could not reach the index) leaves an id
        behind. Read is the only moment we are certain an id is dead — the
        document it points at is gone — so that is where the ZREM happens.
        Without it the index is append-only for the life of the deployment and
        every listing pays to look up ids that will never resolve again.
        """
        owner_key = self._c.keys.owner_index(owner)
        ids = await self._c.redis.zrevrange(owner_key, 0, limit - 1)  # type: ignore[no-untyped-call]
        summaries: list[SessionSummary] = []
        for raw_id in ids:
            sid = SessionId(raw_id.decode() if isinstance(raw_id, bytes) else raw_id)
            sess = await self.get(sid)
            if sess is None:
                await self._c.redis.zrem(owner_key, str(sid))  # type: ignore[no-untyped-call]
                log.info("session_index_entry_pruned", owner=str(owner), session_id=str(sid))
                continue
            summaries.append(
                SessionSummary(
                    id=sess.id,
                    title=sess.title,
                    last_message_at=sess.updated_at,
                    message_count=sess.message_count,
                )
            )
        return summaries

    async def delete(self, session_id: SessionId) -> None:
        sess = await self.get(session_id)
        await self._c.redis.delete(  # type: ignore[no-untyped-call]
            self._c.keys.session(session_id),
            self._c.keys.messages(session_id),
        )
        if sess is not None:
            await self._c.redis.zrem(self._c.keys.owner_index(sess.owner), str(session_id))  # type: ignore[no-untyped-call]

    async def touch(self, session_id: SessionId) -> None:
        """Mark a session active: refresh its timestamp and ALL of its TTLs.

        The messages list is the part that used to be forgotten. A session
        touched every day for a month kept its metadata alive indefinitely
        while its messages key silently ran out its original TTL and vanished —
        the session survived its own history.
        """
        sess = await self.get(session_id)
        if sess is None:
            return
        updated = sess.model_copy(update={"updated_at": datetime.now(UTC)})
        owner_key = self._c.keys.owner_index(updated.owner)
        async with self._c.redis.pipeline(transaction=True) as pipe:  # type: ignore[no-untyped-call]
            pipe.set(  # type: ignore[no-untyped-call]
                self._c.keys.session(updated.id),
                to_versioned_json(updated.model_dump(mode="json"), schema_version=_SCHEMA_V),
                ex=self._ttl,
            )
            # EXPIRE on a missing key is a no-op returning 0 — a session with no
            # messages yet is not an error.
            pipe.expire(self._c.keys.messages(session_id), self._ttl)  # type: ignore[no-untyped-call]
            pipe.zadd(  # type: ignore[no-untyped-call]
                owner_key,
                {str(session_id): updated.updated_at.timestamp() * 1000},
            )
            pipe.expire(owner_key, self._ttl)  # type: ignore[no-untyped-call]
            await pipe.execute()  # type: ignore[no-untyped-call]

    async def _index_owner(self, owner: OwnerId, session_id: SessionId, when: datetime) -> None:
        """Add/refresh a session in its owner's index, and re-arm the index TTL.

        The index expiry rides on the same ``ttl_seconds`` as the documents it
        points at and is pushed forward by every write, so it always outlives
        its members and can never become the one key in the layout that lives
        forever.
        """
        owner_key = self._c.keys.owner_index(owner)
        async with self._c.redis.pipeline(transaction=True) as pipe:  # type: ignore[no-untyped-call]
            pipe.zadd(owner_key, {str(session_id): when.timestamp() * 1000})  # type: ignore[no-untyped-call]
            pipe.expire(owner_key, self._ttl)  # type: ignore[no-untyped-call]
            await pipe.execute()  # type: ignore[no-untyped-call]

    @staticmethod
    def _decode_session(raw: bytes, *, message_count: int) -> Session:
        """Decode a stored document, overriding its advisory message count."""
        data, _ = from_versioned_json(raw)
        data["message_count"] = message_count
        return Session.model_validate(data)

    async def _save_session(self, sess: Session) -> None:
        await self._c.redis.set(  # type: ignore[no-untyped-call]
            self._c.keys.session(sess.id),
            to_versioned_json(sess.model_dump(mode="json"), schema_version=_SCHEMA_V),
            ex=self._ttl,
        )
