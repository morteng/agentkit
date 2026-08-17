"""Approval resolution: every pending call reaches a rendered terminal state.

Covers the ApprovalResolved event on both the human-verdict and the expiry
path, and the taint state a resume must carry across the suspend.
"""

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from agentkit import AgentConfig, AgentSession
from agentkit._ids import CheckpointId, OwnerId, TurnId, new_id
from agentkit.audit import AuditRecord, AuditSink
from agentkit.errors import ApprovalTimeout, CheckpointMissing
from agentkit.events import ApprovalResolved, Errored, TurnEnded
from agentkit.events.approval import UNNAMED_TOOL
from agentkit.guards.taint import TaintSource
from agentkit.loop.context import TurnContext, to_checkpoint_payload
from agentkit.providers.fakes import FakeProvider
from agentkit.store.fakes import FakeCheckpointStore, FakeMemoryStore, FakeSessionStore
from agentkit.tools.registry import ToolRegistry


class _Recorder(AuditSink):
    def __init__(self) -> None:
        self.records: list[AuditRecord] = []

    async def record(self, record: AuditRecord) -> None:
        self.records.append(record)


def _session(audit_sink: AuditSink | None = None) -> AgentSession:
    config = AgentConfig()
    config.stores.session = FakeSessionStore()
    config.stores.memory = FakeMemoryStore()
    config.stores.checkpoint = FakeCheckpointStore()
    return AgentSession(
        owner=OwnerId("u:alice"),
        config=config,
        provider=FakeProvider().script(FakeProvider.text("hi")),
        registry=ToolRegistry(),
        model="m",
        audit_sink=audit_sink,
    )


async def _save_checkpoint(
    session: AgentSession,
    turn_id: TurnId,
    *,
    pending: list[dict],
    timeout_at: datetime,
    taint: list[TaintSource] | None = None,
) -> None:
    ctx = TurnContext.empty()
    ctx.metadata["pending_user_approvals"] = pending
    ctx.metadata["approval_timeout_at"] = timeout_at.isoformat()
    if taint:
        ctx.tainted = True
        ctx.taint_sources = list(taint)
    assert session.config.stores.checkpoint is not None
    await session.config.stores.checkpoint.save(
        CheckpointId(f"approval:{turn_id}"), to_checkpoint_payload(ctx)
    )


@pytest.mark.asyncio
async def test_expiry_emits_a_resolution_per_stranded_call_before_the_error():
    """A timeout that emits nothing is indistinguishable from a hang."""
    session = _session()
    turn_id = new_id(TurnId)
    exc = ApprovalTimeout("approval window expired", call_ids=["c1", "c2"])

    stream = await session._approval_timeout_stream(turn_id, exc)  # pyright: ignore[reportPrivateUsage]
    events = [ev async for ev in stream]

    resolved = [e for e in events if isinstance(e, ApprovalResolved)]
    assert [e.call_id for e in resolved] == ["c1", "c2"]
    for ev in resolved:
        assert ev.expired is True
        # An unanswered approval is not consent.
        assert ev.decision == "deny"
        assert ev.resolved_by == "system"
    # The error still comes, after the per-card events, and the sequence
    # numbers stay monotonic across the whole stream.
    assert isinstance(events[-2], Errored)
    assert isinstance(events[-1], TurnEnded)
    assert [e.sequence for e in events] == list(range(len(events)))


@pytest.mark.asyncio
async def test_timeout_without_known_calls_still_errors():
    session = _session()
    stream = await session._approval_timeout_stream(  # pyright: ignore[reportPrivateUsage]
        new_id(TurnId), ApprovalTimeout("expired")
    )
    events = [ev async for ev in stream]
    assert [type(e).__name__ for e in events] == ["Errored", "TurnEnded"]


def test_resolution_normalises_the_decision_and_drops_irrelevant_fields():
    session = _session()
    ctx = TurnContext.empty()
    turn_id = new_id(TurnId)

    granted = session._build_resolution_event(  # pyright: ignore[reportPrivateUsage]
        ctx, turn_id, "c1", "approve", {"category": "movies"}, "ignored on approve", "u:alice"
    )
    assert granted.decision == "approve"
    assert granted.edited_args == {"category": "movies"}
    assert granted.reason is None

    denied = session._build_resolution_event(  # pyright: ignore[reportPrivateUsage]
        ctx, turn_id, "c1", "deny", {"category": "movies"}, "wrong account", "u:alice"
    )
    assert denied.decision == "deny"
    assert denied.edited_args is None
    assert denied.reason == "wrong account"

    # The session treats anything that is not exactly "approve" as a denial;
    # the event has to report what happened, not what was asked for.
    shouty = session._build_resolution_event(  # pyright: ignore[reportPrivateUsage]
        ctx, turn_id, "c1", "APPROVE", None, None, "u:alice"
    )
    assert shouty.decision == "deny"


def test_unresolved_calls_are_attributed_to_the_system_not_the_human():
    ctx = TurnContext.empty()
    ctx.metadata["pending_user_approvals"] = [
        {"id": "c1", "name": "t", "arguments": {}},
        {"id": "c2", "name": "t", "arguments": {}},
    ]

    verdicts = AgentSession._deny_unresolved(ctx)  # pyright: ignore[reportPrivateUsage]

    assert [v["call_id"] for v in verdicts] == ["c1", "c2"]
    for v in verdicts:
        assert v["decision"] == "deny"
        assert v["resolved_by"] == "system"


@pytest.mark.asyncio
async def test_verdict_pair_is_the_old_event_then_the_new_one():
    """ApprovalGranted stays first for existing consumers; ApprovalResolved
    follows it as the one event a card can close on."""
    session = _session()
    ctx = TurnContext.empty()
    turn_id = new_id(TurnId)
    queue: asyncio.Queue = asyncio.Queue()

    await session._emit_verdict(  # pyright: ignore[reportPrivateUsage]
        queue, ctx, turn_id, "c1", "approve", {"category": "movies"}, None, "u:alice"
    )

    events = [queue.get_nowait() for _ in range(queue.qsize())]
    assert [type(e).__name__ for e in events] == ["ApprovalGranted", "ApprovalResolved"]
    assert events[1].resolved_by == "u:alice"
    assert events[1].call_id == "c1"
    # Both draw from the turn's single sequence counter.
    assert [e.sequence for e in events] == [0, 1]


@pytest.mark.asyncio
async def test_resume_restores_taint_and_its_sources():
    """Dropping taint on resume would re-enable the writes the guard denied."""
    session = _session()
    turn_id = new_id(TurnId)
    await _save_checkpoint(
        session,
        turn_id,
        pending=[{"id": "c1", "name": "t", "arguments": {}}],
        timeout_at=datetime.now(UTC) + timedelta(hours=1),
        taint=[TaintSource(call_id="c0", tool_name="web.fetch", kind="untrusted")],
    )

    ctx, _queue = await session._load_resume_context(turn_id, require_call_ids=["c1"])  # pyright: ignore[reportPrivateUsage]

    assert ctx.tainted is True
    assert ctx.taint_sources == [TaintSource(call_id="c0", tool_name="web.fetch", kind="untrusted")]


@pytest.mark.asyncio
async def test_expired_checkpoint_names_the_calls_it_stranded():
    session = _session()
    await session.initialize()
    turn_id = new_id(TurnId)
    await _save_checkpoint(
        session,
        turn_id,
        pending=[
            {"id": "c1", "name": "t", "arguments": {}},
            {"id": "c2", "name": "t", "arguments": {}},
        ],
        timeout_at=datetime.now(UTC) - timedelta(seconds=1),
    )

    with pytest.raises(ApprovalTimeout) as excinfo:
        await session._load_resume_context(turn_id)  # pyright: ignore[reportPrivateUsage]

    assert excinfo.value.call_ids == ("c1", "c2")


# ---------------------------------------------------------------------------
# K11(b): expiry is detectable on an access path other than resume.
#
# check_approval_expiry is a lazy check, not a sweep — agentkit has no
# background-task lifecycle to drive one. It lets a host that already knows a
# turn_id is awaiting approval (e.g. one it persisted from its own
# ApprovalNeeded event) find out, and close the card out, without a client
# ever having to call resume_with_approval.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_check_approval_expiry_is_none_when_nothing_is_pending():
    session = _session()
    assert await session.check_approval_expiry(new_id(TurnId)) is None


@pytest.mark.asyncio
async def test_check_approval_expiry_leaves_a_live_checkpoint_untouched():
    """Not yet expired: this is a read, not a destructive peek at something
    still answerable."""
    session = _session()
    turn_id = new_id(TurnId)
    await _save_checkpoint(
        session,
        turn_id,
        pending=[{"id": "c1", "name": "t", "arguments": {}}],
        timeout_at=datetime.now(UTC) + timedelta(hours=1),
    )

    assert await session.check_approval_expiry(turn_id) is None

    # A real resume still finds it — nothing was consumed.
    ctx, _queue = await session._load_resume_context(  # pyright: ignore[reportPrivateUsage]
        turn_id, require_call_ids=["c1"]
    )
    assert ctx.turn_id == turn_id


@pytest.mark.asyncio
async def test_check_approval_expiry_closes_out_an_approval_nobody_ever_resumed():
    """The core K11(b) case: no client ever calls resume. This is the only
    other thing that can discover the expiry, and it must produce the same
    outcome resume's timeout path would have."""
    audit = _Recorder()
    session = _session(audit)
    turn_id = new_id(TurnId)
    await _save_checkpoint(
        session,
        turn_id,
        pending=[
            {"id": "c1", "name": "t", "arguments": {}},
            {"id": "c2", "name": "t", "arguments": {}},
        ],
        timeout_at=datetime.now(UTC) - timedelta(seconds=1),
    )

    stream = await session.check_approval_expiry(turn_id)
    assert stream is not None
    events = [ev async for ev in stream]

    resolved = [e for e in events if isinstance(e, ApprovalResolved)]
    assert [e.call_id for e in resolved] == ["c1", "c2"]
    for ev in resolved:
        assert ev.expired is True
        assert ev.decision == "deny"
        assert ev.resolved_by == "system"
    assert isinstance(events[-2], Errored)
    assert isinstance(events[-1], TurnEnded)
    assert [e.sequence for e in events] == list(range(len(events)))

    # Audited, same as the resume timeout path. (This case drains the stream;
    # the "nobody drains it" guarantee is covered separately below — the
    # comment that used to be here claimed it and did not test it.)
    assert [r.call_id for r in audit.records] == ["c1", "c2"]
    for record in audit.records:
        assert record.detail["expired"] is True
        assert record.detail["decision"] == "deny"

    # Consumed: a later resume finds nothing to resume.
    with pytest.raises(CheckpointMissing):
        await session._load_resume_context(turn_id)  # pyright: ignore[reportPrivateUsage]


# ---------------------------------------------------------------------------
# The audit row for an expiry does not depend on anyone watching the stream.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_expiry_is_audited_even_when_the_client_never_reads_the_stream():
    """FAILS PRE-FIX (0 audit records instead of 1).

    The audit loop used to sit at the top of the async generator that yields
    the timeout events. An async generator body does not execute until its
    first ``__anext__``, so a consumer that entered ``resume_with_approval``'s
    context manager and then simply closed its approval card — without
    iterating — produced no audit row at all, while the comment above the loop
    claimed the write happened "whether or not the consumer drains this
    stream".

    This drives the real seam: the public ``resume_with_approval`` context
    manager, abandoned exactly the way a UI abandons it. Asserting on
    ``_approval_timeout_stream`` directly would not have caught it, because
    every existing test of that helper drains what it returns.
    """
    audit = _Recorder()
    session = _session(audit)
    turn_id = new_id(TurnId)
    await _save_checkpoint(
        session,
        turn_id,
        pending=[{"id": "c1", "name": "t", "arguments": {}}],
        timeout_at=datetime.now(UTC) - timedelta(seconds=1),
    )

    async with session.resume_with_approval(turn_id, "c1", decision="approve"):
        pass  # never iterated — the card was just dismissed

    assert [r.call_id for r in audit.records] == ["c1"]
    assert audit.records[0].detail["expired"] is True
    assert audit.records[0].detail["decision"] == "deny"
    # A verdict the runtime reached on its own is agent activity, never filed
    # as a human decision.
    assert audit.records[0].actor == "system"


@pytest.mark.asyncio
async def test_an_abandoned_expiry_stream_still_audits_exactly_once():
    """Writing the rows eagerly must not also write them again on iteration.

    Guards the obvious wrong fix for the test above — leaving the loop inside
    the generator and adding a second one outside it. Passes pre-fix too (the
    pre-fix code audits exactly once *when* drained); it is here to pin the
    "exactly once" half that the eager write could regress.
    """
    audit = _Recorder()
    session = _session(audit)
    turn_id = new_id(TurnId)
    await _save_checkpoint(
        session,
        turn_id,
        pending=[
            {"id": "c1", "name": "t", "arguments": {}},
            {"id": "c2", "name": "t", "arguments": {}},
        ],
        timeout_at=datetime.now(UTC) - timedelta(seconds=1),
    )

    async with session.resume_with_approval(turn_id, "c1", decision="approve") as stream:
        events = [ev async for ev in stream]

    assert [e.call_id for e in events if isinstance(e, ApprovalResolved)] == ["c1", "c2"]
    assert [r.call_id for r in audit.records] == ["c1", "c2"]  # once each, not twice


# --- the tool name on the outcome, not just on the request -------------------
#
# ApprovalNeeded has always carried tool_name; every event reporting how the
# approval *ended* dropped it, so a consumer holding a verdict could not say
# what it had authorised. These drive the real builders rather than
# constructing events directly — the field existing proves nothing, the
# question is whether the name survives the path.


@pytest.mark.asyncio
async def test_expiry_names_the_tool_even_though_the_bucket_is_cleared():
    """The hard case: expiry empties pending_user_approvals, then builds events.

    _close_expired_checkpoint clears the bucket the names live in, and
    _approval_timeout_stream runs afterwards — so unless the names ride along
    on the exception they are already gone. An expired approval used to render
    and audit as a bare call id.
    """
    recorder = _Recorder()
    session = _session(recorder)
    turn_id = new_id(TurnId)
    ctx = TurnContext.empty()
    ctx.metadata["pending_user_approvals"] = [
        {"id": "c1", "name": "torrent_admin.torrent_delete", "arguments": {}},
        {"id": "c2", "name": "music_retag", "arguments": {}},
    ]

    exc = await session._close_expired_checkpoint(  # pyright: ignore[reportPrivateUsage]
        ctx, turn_id, "2026-08-17T00:00:00+00:00"
    )
    # Precondition for the whole test: the source of the names is now empty.
    assert ctx.metadata["pending_user_approvals"] == []

    stream = await session._approval_timeout_stream(turn_id, exc)  # pyright: ignore[reportPrivateUsage]
    events = [ev async for ev in stream]

    resolved = [e for e in events if isinstance(e, ApprovalResolved)]
    assert {e.call_id: e.tool_name for e in resolved} == {
        "c1": "torrent_admin.torrent_delete",
        "c2": "music_retag",
    }
    # And the audit rows, which hardcoded "" on this path.
    expired_rows = {
        r.call_id: r.target for r in recorder.records if r.action == "approval_resolved"
    }
    assert expired_rows == {"c1": "torrent_admin.torrent_delete", "c2": "music_retag"}


@pytest.mark.asyncio
async def test_expiry_falls_back_to_unnamed_for_a_call_it_never_saw():
    """Control: the mapping can miss, and the miss is visible as UNNAMED_TOOL.

    Without this, the test above would pass against an implementation that
    returned a constant, and "" would read as success rather than as absence.
    """
    session = _session()
    exc = ApprovalTimeout("expired", call_ids=["c1"], tool_names={"other": "music_retag"})

    stream = await session._approval_timeout_stream(new_id(TurnId), exc)  # pyright: ignore[reportPrivateUsage]
    resolved = [e async for e in stream if isinstance(e, ApprovalResolved)]

    assert [e.tool_name for e in resolved] == [UNNAMED_TOOL]


def test_verdict_and_resolution_events_name_the_tool():
    session = _session()
    ctx = TurnContext.empty()
    turn_id = new_id(TurnId)
    # Post-verdict shape: _apply_approval_decision has already moved the entry
    # out of pending_user_approvals, which is why the lookup searches all three.
    ctx.metadata["approved_tool_calls"] = [
        {"id": "c1", "name": "music_retag", "arguments": {}},
    ]
    ctx.metadata["denied_tool_calls"] = [
        {"id": "c2", "name": "torrent_admin.torrent_delete", "arguments": {}},
    ]

    granted = session._build_verdict_event(ctx, turn_id, "c1", "approve", None, None)  # pyright: ignore[reportPrivateUsage]
    assert granted.tool_name == "music_retag"

    denied = session._build_verdict_event(ctx, turn_id, "c2", "deny", None, "no")  # pyright: ignore[reportPrivateUsage]
    assert denied.tool_name == "torrent_admin.torrent_delete"

    resolved = session._build_resolution_event(  # pyright: ignore[reportPrivateUsage]
        ctx, turn_id, "c1", "approve", None, None, "u:alice"
    )
    assert resolved.tool_name == "music_retag"

    # Control: a call in none of the buckets is reported as unnamed rather
    # than borrowing whichever name happened to be first.
    stray = session._build_resolution_event(  # pyright: ignore[reportPrivateUsage]
        ctx, turn_id, "c-unknown", "deny", None, None, "u:alice"
    )
    assert stray.tool_name == UNNAMED_TOOL
