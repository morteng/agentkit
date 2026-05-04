"""Redis-backed SessionStore.

Layout:
- ``{prefix}:sess:{id}``               JSON session metadata
- ``{prefix}:msgs:{id}``               LIST of message JSONs (RPUSH on append)
- ``{prefix}:owner:{owner}:sessions``  ZSET (score = updated_at unix ms) of session ids

TTL: each session and its messages list expire after ``ttl_seconds`` of inactivity
(default 30 days). ``touch`` extends the TTL.
"""

from datetime import UTC, datetime
from typing import cast

from agentkit._ids import OwnerId, SessionId
from agentkit._messages import Message
from agentkit.errors import StoreError
from agentkit.store.redis.client import RedisClient
from agentkit.store.redis.serialization import from_versioned_json, to_versioned_json
from agentkit.store.session import Session, SessionStore, SessionSummary

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
        await self._c.redis.zadd(  # type: ignore[no-untyped-call]
            self._c.keys.owner_index(owner),
            {str(session_id): now.timestamp() * 1000},
        )
        return sess

    async def get(self, session_id: SessionId) -> Session | None:
        raw = await self._c.redis.get(self._c.keys.session(session_id))  # type: ignore[no-untyped-call]
        if raw is None:
            return None
        assert isinstance(raw, bytes)
        data, _ = from_versioned_json(raw)
        return Session.model_validate(data)

    async def append_message(self, session_id: SessionId, message: Message) -> None:
        sess = await self.get(session_id)
        if sess is None:
            raise StoreError(f"session not found: {session_id}")
        await self._c.redis.rpush(  # type: ignore[no-untyped-call]
            self._c.keys.messages(session_id),
            to_versioned_json(message.model_dump(mode="json"), schema_version=_SCHEMA_V),
        )
        await self._c.redis.expire(self._c.keys.messages(session_id), self._ttl)  # type: ignore[no-untyped-call]
        updated = sess.model_copy(
            update={
                "message_count": sess.message_count + 1,
                "updated_at": datetime.now(UTC),
            }
        )
        await self._save_session(updated)
        await self._c.redis.zadd(  # type: ignore[no-untyped-call]
            self._c.keys.owner_index(updated.owner),
            {str(session_id): updated.updated_at.timestamp() * 1000},
        )

    async def list_messages(self, session_id: SessionId, *, limit: int = 200) -> list[Message]:
        raws: list[bytes] = await self._c.redis.lrange(  # type: ignore[no-untyped-call,reportUnknownVariableType]
            self._c.keys.messages(session_id), -limit, -1
        )
        return [
            Message.model_validate(from_versioned_json(cast("bytes", r))[0])
            for r in raws  # type: ignore[reportUnknownVariableType]
        ]

    async def list_for_owner(self, owner: OwnerId, *, limit: int = 30) -> list[SessionSummary]:
        ids = await self._c.redis.zrevrange(self._c.keys.owner_index(owner), 0, limit - 1)  # type: ignore[no-untyped-call]
        summaries: list[SessionSummary] = []
        for raw_id in ids:
            sid = SessionId(raw_id.decode() if isinstance(raw_id, bytes) else raw_id)
            sess = await self.get(sid)
            if sess is None:
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
        sess = await self.get(session_id)
        if sess is None:
            return
        updated = sess.model_copy(update={"updated_at": datetime.now(UTC)})
        await self._save_session(updated)
        await self._c.redis.zadd(  # type: ignore[no-untyped-call]
            self._c.keys.owner_index(updated.owner),
            {str(session_id): updated.updated_at.timestamp() * 1000},
        )

    async def _save_session(self, sess: Session) -> None:
        await self._c.redis.set(  # type: ignore[no-untyped-call]
            self._c.keys.session(sess.id),
            to_versioned_json(sess.model_dump(mode="json"), schema_version=_SCHEMA_V),
            ex=self._ttl,
        )
