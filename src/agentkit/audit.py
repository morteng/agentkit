"""Audit sink — the durable record of what the agent actually did.

An audit trail a caller has to remember to wire up is an audit trail with holes
in it, and the holes land where they hurt most. The pattern this module
replaces threaded a per-client ``audit=`` callback through every tool client:
the torrent client got one and writes a row for every download, the Vaultwarden
client did not, so permanently destroying a family member's password vault left
no record at all. Nothing in that design made the omission visible — it was one
missing keyword argument in one constructor.

So the sink lives on the :class:`~agentkit.session.AgentSession` and the
runtime emits from the two places every action must pass through:

* :class:`~agentkit.loop.tool_dispatcher.ToolDispatcher` — one record per tool
  call, whatever the tool, whoever wrote it, however it was registered;
* :meth:`~agentkit.session.AgentSession._emit_verdict` and the expiry path —
  one record per approval that reached a terminal state, because *who
  consented* is the half of a destructive action a tool record cannot show.

A consumer that configures nothing gets :class:`NullAuditSink` and the same
silence as before. What is no longer possible is a *partially* audited runtime,
where the answer to "is this tool audited?" depends on which client someone
wired.

Attach several sinks with :class:`MultiAuditSink` — the usual shape is a store
writer plus a chat mirror, and one of them being down must not lose the other's
row.
"""

from collections.abc import Sequence
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from agentkit._logging import get_logger

log = get_logger(__name__)

#: ``AuditRecord.action`` for one dispatched tool call.
ACTION_TOOL_CALL = "tool_call"

#: ``AuditRecord.action`` for an approval that reached a terminal state.
ACTION_APPROVAL_RESOLVED = "approval_resolved"

#: ``AuditRecord.source`` for something the agent did on a principal's behalf,
#: including a verdict the runtime reached on its own (an expiry). Visible to
#: everyone in a household deployment.
SOURCE_AGENT = "agent"

#: ``AuditRecord.source`` for a human's own decision — an approval verdict a
#: person actually gave. A human's consent must never be filed as agent
#: activity: it is the one row that proves the loop had a human in it.
SOURCE_HUMAN = "human"


class AuditRecord(BaseModel):
    """One thing that happened, in a shape a ledger row can take verbatim.

    The first six fields are the whole record for a consumer that just wants a
    log; the id fields let one that keeps transcripts join a row back to the
    turn that produced it.
    """

    ts: datetime
    actor: str
    """Who is answerable: the session owner for a tool call, the resolver for an
    approval verdict (``"system"`` when the runtime ruled on its own)."""
    action: str
    target: str = ""
    """What was acted on — the tool name for a tool call. Empty only when the
    runtime genuinely does not know it; ``call_id`` still identifies the call.

    Expiry used to be exactly that case: the checkpoint holding the pending
    call was cleared before these rows were written, so every expired approval
    audited as a bare id. The names now travel on
    :class:`~agentkit.errors.ApprovalTimeout` and expired rows are named like
    any other."""
    detail: dict[str, Any] = Field(default_factory=dict)
    """Already projected for reading: argument previews are flattened and
    secret-looking keys masked (:mod:`agentkit._redaction`). Never a place to
    put a full payload back."""
    source: str = SOURCE_AGENT
    session_id: str | None = None
    turn_id: str | None = None
    call_id: str | None = None


@runtime_checkable
class AuditSink(Protocol):
    """Somewhere an :class:`AuditRecord` is written.

    Implementations should be cheap and should not raise; both the runtime call
    sites go through :func:`record_audit`, which swallows and logs whatever
    escapes, because a failed audit write must not also destroy the turn.
    """

    async def record(self, record: AuditRecord) -> None: ...


class NullAuditSink(AuditSink):
    """The default: writes nothing.

    Deliberately a real object rather than ``None``. The runtime always has a
    sink and always calls it, so wiring a real one is a single assignment on the
    config, and no call site ever has to ask whether auditing is on.
    """

    async def record(self, record: AuditRecord) -> None:
        return None


class LoggingAuditSink(AuditSink):
    """Writes each record to the structured log. Enough to start with."""

    def __init__(self, *, event: str = "agentkit.audit") -> None:
        self._event = event

    async def record(self, record: AuditRecord) -> None:
        log.info(self._event, **record.model_dump(mode="json"))


class MultiAuditSink(AuditSink):
    """Fan one record out to several sinks, isolating their failures.

    A store writer and a chat mirror have very different reliability, and the
    mirror is the flaky one. Isolating per sink means the durable row still
    lands when the mirror is down — the opposite arrangement loses the record
    that mattered to protect the one that did not.
    """

    def __init__(self, sinks: Sequence[AuditSink]) -> None:
        self._sinks = list(sinks)

    async def record(self, record: AuditRecord) -> None:
        for sink in self._sinks:
            try:
                await sink.record(record)
            except Exception:
                log.exception(
                    "audit_sink_failed",
                    sink=type(sink).__name__,
                    action=record.action,
                    target=record.target,
                )


async def record_audit(sink: AuditSink | None, record: AuditRecord) -> None:
    """Write ``record`` to ``sink``. Never raises.

    The action being audited has already happened by the time this runs — the
    tool executed, the human clicked approve. Letting a sink failure propagate
    would lose the record *and* corrupt the outcome of a turn that otherwise
    succeeded. The record is logged at error level with its full contents
    instead, so it is recoverable from logs rather than gone.
    """
    if sink is None:
        return
    try:
        await sink.record(record)
    except Exception:
        log.exception("audit_write_failed", record=record.model_dump(mode="json"))
