"""Session store protocol + Session/SessionSummary types."""

from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from agentkit._ids import OwnerId, SessionId
from agentkit._messages import Message


class Session(BaseModel):
    id: SessionId
    owner: OwnerId
    title: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)  # type: ignore[reportUnknownVariableType]
    created_at: datetime
    updated_at: datetime
    message_count: int = 0
    config_overrides: dict[str, Any] = Field(default_factory=dict)  # type: ignore[reportUnknownVariableType]


class SessionSummary(BaseModel):
    id: SessionId
    title: str | None
    last_message_at: datetime
    message_count: int


@runtime_checkable
class SessionStore(Protocol):
    """Conversation history + per-session state."""

    async def create(
        self,
        session_id: SessionId,
        owner: OwnerId,
        metadata: dict[str, str] | None = None,
        title: str | None = None,
    ) -> Session: ...

    async def get(self, session_id: SessionId) -> Session | None: ...

    async def append_message(self, session_id: SessionId, message: Message) -> None: ...

    async def list_messages(
        self,
        session_id: SessionId,
        *,
        limit: int = 200,
    ) -> list[Message]: ...

    async def list_for_owner(
        self,
        owner: OwnerId,
        *,
        limit: int = 30,
    ) -> list[SessionSummary]: ...

    async def delete(self, session_id: SessionId) -> None: ...

    async def touch(self, session_id: SessionId) -> None: ...
