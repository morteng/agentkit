"""AgentSession.expire_due — the sweep for approvals nobody ever answers.

K11(b). ``check_approval_expiry(turn_id)`` already closed out an approval whose
turn the caller could name; every test here is about the case where nobody can
name it, which is the case "silence is not consent" actually depends on. An
approval that expires unswept emits no ApprovalResolved and writes no audit
row, so the card stays on screen and the record says the question was never
answered rather than that it was answered by running out the clock.

Every test in this file fails pre-fix with ``AttributeError: 'AgentSession'
object has no attribute 'expire_due'`` unless its docstring says otherwise.
"""

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from agentkit import AgentConfig, AgentSession
from agentkit._ids import CheckpointId, OwnerId, SessionId, TurnId, new_id
from agentkit.audit import AuditRecord, AuditSink
from agentkit.errors import CheckpointMissing
from agentkit.events import ApprovalResolved, Errored, TurnEnded
from agentkit.loop.context import TurnContext, to_checkpoint_payload
from agentkit.providers.fakes import FakeProvider
from agentkit.store.checkpoint import CheckpointPayload, approval_checkpoint_id
from agentkit.store.fakes import FakeCheckpointStore, FakeMemoryStore, FakeSessionStore
from agentkit.tools.registry import ToolRegistry


class _Recorder(AuditSink):
    def __init__(self) -> None:
        self.records: list[AuditRecord] = []

    async def record(self, record: AuditRecord) -> None:
        self.records.append(record)


class _BlindCheckpointStore:
    """A CheckpointStore with no ``list_ids`` — the pre-upgrade third-party shape.

    Deliberately not a subclass of FakeCheckpointStore: the point is a store
    that satisfies the published ``CheckpointStore`` protocol and nothing more,
    which is what every host implementation written before ``list_ids`` existed
    looks like.
    """

    def __init__(self) -> None:
        self._data: dict[CheckpointId, CheckpointPayload] = {}

    async def save(self, checkpoint_id: CheckpointId, payload: CheckpointPayload) -> None:
        self._data[checkpoint_id] = payload

    async def load(self, checkpoint_id: CheckpointId) -> CheckpointPayload | None:
        return self._data.get(checkpoint_id)

    async def delete(self, checkpoint_id: CheckpointId) -> None:
        self._data.pop(checkpoint_id, None)


def _session(audit_sink: AuditSink | None = None, *, checkpoint: Any = None) -> AgentSession:
    config = AgentConfig()
    config.stores.session = FakeSessionStore()
    config.stores.memory = FakeMemoryStore()
    config.stores.checkpoint = FakeCheckpointStore() if checkpoint is None else checkpoint
    return AgentSession(
        owner=OwnerId("u:alice"),
        config=config,
        provider=FakeProvider().script(FakeProvider.text("hi")),
        registry=ToolRegistry(),
        model="m",
        audit_sink=audit_sink,
    )


async def _suspend(
    session: AgentSession,
    *,
    pending: list[dict[str, Any]],
    expired: bool,
    session_id: SessionId | None = None,
) -> TurnId:
    """Write the checkpoint an approval suspend would have left behind.

    ``session_id`` defaults to the session's own, which is what the real
    approval_wait handler writes; the cross-session test overrides it.
    """
    turn_id = new_id(TurnId)
    ctx = TurnContext(
        session_id=session_id if session_id is not None else session.id,
        turn_id=turn_id,
        call_id="",
        history=[],
    )
    ctx.metadata["pending_user_approvals"] = pending
    delta = timedelta(seconds=-1) if expired else timedelta(hours=1)
    ctx.metadata["approval_timeout_at"] = (datetime.now(UTC) + delta).isoformat()
    assert session.config.stores.checkpoint is not None
    await session.config.stores.checkpoint.save(
        approval_checkpoint_id(turn_id), to_checkpoint_payload(ctx)
    )
    return turn_id


@pytest.mark.asyncio
async def test_expire_due_closes_an_approval_no_client_ever_came_back_for():
    """The whole point of K11(b): nothing but the sweep ever names this turn."""
    audit = _Recorder()
    session = _session(audit)
    turn_id = await _suspend(
        session,
        pending=[
            {"id": "c1", "name": "vault.delete", "arguments": {}},
            {"id": "c2", "name": "vault.delete", "arguments": {}},
        ],
        expired=True,
    )

    events = await session.expire_due()

    resolved = [e for e in events if isinstance(e, ApprovalResolved)]
    assert [e.call_id for e in resolved] == ["c1", "c2"]
    for ev in resolved:
        assert ev.expired is True
        assert ev.decision == "deny"  # an unanswered approval is not consent
        assert ev.resolved_by == "system"
        assert ev.turn_id == turn_id
    assert isinstance(events[-2], Errored)
    assert isinstance(events[-1], TurnEnded)

    # The audit row is the half a tool-call record cannot show.
    assert [r.call_id for r in audit.records] == ["c1", "c2"]
    for record in audit.records:
        assert record.detail["expired"] is True
        assert record.detail["decision"] == "deny"
        assert record.actor == "system"

    # Consumed: the checkpoint is gone.
    assert session.config.stores.checkpoint is not None
    assert await session.config.stores.checkpoint.load(approval_checkpoint_id(turn_id)) is None


@pytest.mark.asyncio
async def test_expire_due_leaves_a_still_answerable_approval_alone():
    """A sweep must not be a destructive peek. The window is still open, so the
    approval stays exactly as resumable as it was."""
    session = _session()
    turn_id = await _suspend(
        session, pending=[{"id": "c1", "name": "t", "arguments": {}}], expired=False
    )

    assert await session.expire_due() == []

    ctx, _queue = await session._load_resume_context(  # pyright: ignore[reportPrivateUsage]
        turn_id, require_call_ids=["c1"]
    )
    assert ctx.turn_id == turn_id


@pytest.mark.asyncio
async def test_expire_due_never_closes_another_sessions_approval():
    """Checkpoint ids are keyed by turn, not by session, so the sweep sees
    every session's pending approvals in the shared store. Closing a foreign
    one from here would emit ApprovalResolved under this session's id, persist
    the stranded turn into this session's message store, and file the audit row
    against the wrong session — three quiet corruptions replacing one missing
    event."""
    audit = _Recorder()
    session = _session(audit)
    foreign = new_id(SessionId)
    foreign_turn = await _suspend(
        session,
        pending=[{"id": "c1", "name": "t", "arguments": {}}],
        expired=True,
        session_id=foreign,
    )

    assert await session.expire_due() == []
    assert audit.records == []
    # Untouched, not merely unreported: the other session can still close it.
    assert session.config.stores.checkpoint is not None
    assert await session.config.stores.checkpoint.load(approval_checkpoint_id(foreign_turn))


@pytest.mark.asyncio
async def test_expire_due_and_a_later_resume_do_not_both_fire():
    """A client that wakes up after the sweep must not get a second
    resolution — the card was already closed and the audit already written."""
    audit = _Recorder()
    session = _session(audit)
    turn_id = await _suspend(
        session, pending=[{"id": "c1", "name": "t", "arguments": {}}], expired=True
    )

    swept = await session.expire_due()
    assert len([e for e in swept if isinstance(e, ApprovalResolved)]) == 1
    assert len(audit.records) == 1

    with pytest.raises(CheckpointMissing):
        await session._load_resume_context(turn_id, require_call_ids=["c1"])  # pyright: ignore[reportPrivateUsage]

    # And no second audit row appeared on the way to that exception.
    assert len(audit.records) == 1


@pytest.mark.asyncio
async def test_sweeping_twice_resolves_and_audits_exactly_once():
    """Idempotence falls out of the checkpoint delete, but only if the sweep
    really does delete before it reports."""
    audit = _Recorder()
    session = _session(audit)
    await _suspend(session, pending=[{"id": "c1", "name": "t", "arguments": {}}], expired=True)

    first = await session.expire_due()
    second = await session.expire_due()

    assert len([e for e in first if isinstance(e, ApprovalResolved)]) == 1
    assert second == []
    assert len(audit.records) == 1


@pytest.mark.asyncio
async def test_expire_due_sweeps_every_overdue_turn_in_one_pass():
    """Drives more than one turn rather than a sample: the sweep loop is the
    thing under test, and a one-turn test passes against a body that breaks
    after the first iteration."""
    session = _session()
    expired_turns = {
        await _suspend(
            session, pending=[{"id": f"c{i}", "name": "t", "arguments": {}}], expired=True
        )
        for i in range(3)
    }
    live_turn = await _suspend(
        session, pending=[{"id": "live", "name": "t", "arguments": {}}], expired=False
    )

    events = await session.expire_due()

    resolved = [e for e in events if isinstance(e, ApprovalResolved)]
    assert {e.turn_id for e in resolved} == expired_turns
    assert "live" not in {e.call_id for e in resolved}
    assert session.config.stores.checkpoint is not None
    assert await session.config.stores.checkpoint.load(approval_checkpoint_id(live_turn))


@pytest.mark.asyncio
async def test_expire_due_ignores_checkpoints_that_are_not_approvals():
    """The store is shared with anything else the host persists mid-turn. A
    sweep that tried to read those as approval checkpoints would either crash
    on the JSON or invent a resolution for something that was never an
    approval."""
    session = _session()
    assert session.config.stores.checkpoint is not None
    await session.config.stores.checkpoint.save(
        CheckpointId("some-other-subsystem:42"), b"not json at all"
    )

    assert await session.expire_due() == []


@pytest.mark.asyncio
async def test_expire_due_refuses_a_store_it_cannot_enumerate():
    """A sweep that cannot see the keyspace must not report "nothing was due".

    That answer is indistinguishable from a real all-clear, and believing it is
    exactly the silent-consent failure the sweep exists to remove — so a store
    that predates ``list_ids`` is a loud configuration error, not a quiet
    no-op.
    """
    session = _session(checkpoint=_BlindCheckpointStore())
    await _suspend(session, pending=[{"id": "c1", "name": "t", "arguments": {}}], expired=True)

    with pytest.raises(TypeError, match="EnumerableCheckpointStore"):
        await session.expire_due()


@pytest.mark.asyncio
async def test_expire_due_is_quietly_empty_without_a_checkpoint_store():
    """Distinct from the non-enumerable case: no checkpoint store at all means
    approvals cannot suspend, so nothing can have expired. That is a real
    all-clear, not a blind one."""
    session = _session()
    session.config.stores.checkpoint = None

    assert await session.expire_due() == []
