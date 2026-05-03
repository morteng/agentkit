"""In-memory FakeSessionStore. Production code path: tests + dev only."""

from datetime import UTC, datetime

from agentkit._ids import OwnerId, SessionId
from agentkit._messages import Message
from agentkit.store.session import Session, SessionStore, SessionSummary


class FakeSessionStore(SessionStore):
    def __init__(self) -> None:
        self._sessions: dict[SessionId, Session] = {}
        self._messages: dict[SessionId, list[Message]] = {}

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
        self._sessions[session_id] = sess
        self._messages[session_id] = []
        return sess

    async def get(self, session_id: SessionId) -> Session | None:
        return self._sessions.get(session_id)

    async def append_message(self, session_id: SessionId, message: Message) -> None:
        self._messages.setdefault(session_id, []).append(message)
        if (sess := self._sessions.get(session_id)) is not None:
            sess = sess.model_copy(
                update={
                    "message_count": sess.message_count + 1,
                    "updated_at": datetime.now(UTC),
                }
            )
            self._sessions[session_id] = sess

    async def list_messages(self, session_id: SessionId, *, limit: int = 200) -> list[Message]:
        return list(self._messages.get(session_id, [])[-limit:])

    async def list_for_owner(self, owner: OwnerId, *, limit: int = 30) -> list[SessionSummary]:
        sessions = sorted(
            (s for s in self._sessions.values() if s.owner == owner),
            key=lambda s: s.updated_at,
            reverse=True,
        )[:limit]
        return [
            SessionSummary(
                id=s.id,
                title=s.title,
                last_message_at=s.updated_at,
                message_count=s.message_count,
            )
            for s in sessions
        ]

    async def delete(self, session_id: SessionId) -> None:
        self._sessions.pop(session_id, None)
        self._messages.pop(session_id, None)

    async def touch(self, session_id: SessionId) -> None:
        if (sess := self._sessions.get(session_id)) is not None:
            self._sessions[session_id] = sess.model_copy(update={"updated_at": datetime.now(UTC)})
