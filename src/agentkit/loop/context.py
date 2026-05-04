"""TurnContext — per-turn mutable state passed to handlers and tools.

Grows over the project. v0.1 carries: identifiers, message history, tool calls,
finalize flag, scratchpad, clock, an event queue (later), guards (later),
and a result-cache pointer (later).
"""

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol

from agentkit._ids import SessionId, TurnId, new_id
from agentkit._messages import Message
from agentkit.store.memory import MemoryScope, MemoryStore

if TYPE_CHECKING:
    import asyncio


class Clock(Protocol):
    def now(self) -> datetime: ...


class SystemClock(Clock):
    def now(self) -> datetime:
        return datetime.now(UTC)


@dataclass(frozen=True)
class FixedClock(Clock):
    fixed: datetime

    def now(self) -> datetime:
        return self.fixed


@dataclass
class TurnContext:
    """Per-turn mutable state."""

    session_id: SessionId
    turn_id: TurnId
    call_id: str  # current tool-call id (set by dispatcher)
    history: list[Message] = field(default_factory=list)  # type: ignore[reportUnknownVariableType]
    scratchpad: list[str] = field(default_factory=list)  # type: ignore[reportUnknownVariableType]
    finalize_called: bool = False
    finalize_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)  # type: ignore[reportUnknownVariableType]
    clock: Clock = field(default_factory=SystemClock)

    # Memory + event delivery + approval queue (filled in by Loop, not constructed manually).
    memory_store: MemoryStore | None = None
    memory_scope: MemoryScope | None = None
    event_queue: "asyncio.Queue[Any] | None" = None
    pending_approvals: list[Any] = field(default_factory=list)  # type: ignore[reportUnknownVariableType]
    phase_log: list[tuple[str, str, int]] = field(default_factory=list)  # type: ignore[reportUnknownVariableType]
    """List of (from_phase, to_phase, duration_ms). Populated by the orchestrator."""
    spawn_subagent: Any | None = None  # callable injected by Loop; signature: see subagent.py

    @classmethod
    def empty(
        cls,
        *,
        call_id: str = "",
        clock: Clock | None = None,
        memory_store: MemoryStore | None = None,
        memory_scope: MemoryScope | None = None,
    ) -> "TurnContext":
        return cls(
            session_id=new_id(SessionId),
            turn_id=new_id(TurnId),
            call_id=call_id,
            clock=clock or SystemClock(),
            memory_store=memory_store,
            memory_scope=memory_scope,
        )

    def add_message(self, msg: Message) -> None:
        self.history.append(msg)

    def add_messages(self, msgs: Iterable[Message]) -> None:
        self.history.extend(msgs)


def to_checkpoint_payload(ctx: TurnContext) -> bytes:
    """Serialise the resumable subset of a TurnContext to bytes.

    We don't try to round-trip the entire dataclass — only the fields needed
    to restart the Loop at TOOL_EXECUTING after user approval. Stores, queues,
    and clocks are non-portable and rebuilt by the resuming AgentSession.
    """
    payload: dict[str, Any] = {
        "session_id": ctx.session_id,
        "turn_id": ctx.turn_id,
        "history": [m.model_dump(mode="json") for m in ctx.history],
        "scratchpad": list(ctx.scratchpad),
        "metadata": {
            k: v
            for k, v in ctx.metadata.items()
            if k != "owner"  # rebuilt from session
        },
    }
    return json.dumps(payload, default=str).encode("utf-8")


def from_checkpoint_payload(data: bytes) -> dict[str, Any]:
    """Return a dict that the resuming Loop wires back into a fresh TurnContext."""
    return json.loads(data)
