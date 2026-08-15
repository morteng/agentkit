"""Approval resolution: every pending call reaches a rendered terminal state.

Covers the ApprovalResolved event on both the human-verdict and the expiry
path, and the taint state a resume must carry across the suspend.
"""

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from agentkit import AgentConfig, AgentSession
from agentkit._ids import CheckpointId, OwnerId, TurnId, new_id
from agentkit.errors import ApprovalTimeout
from agentkit.events import ApprovalResolved, Errored, TurnEnded
from agentkit.guards.taint import TaintSource
from agentkit.loop.context import TurnContext, to_checkpoint_payload
from agentkit.providers.fakes import FakeProvider
from agentkit.store.fakes import FakeCheckpointStore, FakeMemoryStore, FakeSessionStore
from agentkit.tools.registry import ToolRegistry


def _session() -> AgentSession:
    config = AgentConfig()
    config.stores.session = FakeSessionStore()
    config.stores.memory = FakeMemoryStore()
    config.stores.checkpoint = FakeCheckpointStore()
    return AgentSession(
        owner=OwnerId("u:morten"),
        config=config,
        provider=FakeProvider().script(FakeProvider.text("hi")),
        registry=ToolRegistry(),
        model="m",
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


def test_expiry_emits_a_resolution_per_stranded_call_before_the_error():
    """A timeout that emits nothing is indistinguishable from a hang."""
    session = _session()
    turn_id = new_id(TurnId)
    exc = ApprovalTimeout("approval window expired", call_ids=["c1", "c2"])

    events = _collect(session._approval_timeout_stream(turn_id, exc))  # pyright: ignore[reportPrivateUsage]

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


def test_timeout_without_known_calls_still_errors():
    session = _session()
    events = _collect(
        session._approval_timeout_stream(new_id(TurnId), ApprovalTimeout("expired"))  # pyright: ignore[reportPrivateUsage]
    )
    assert [type(e).__name__ for e in events] == ["Errored", "TurnEnded"]


def test_resolution_normalises_the_decision_and_drops_irrelevant_fields():
    session = _session()
    ctx = TurnContext.empty()
    turn_id = new_id(TurnId)

    granted = session._build_resolution_event(  # pyright: ignore[reportPrivateUsage]
        ctx, turn_id, "c1", "approve", {"category": "movies"}, "ignored on approve", "u:morten"
    )
    assert granted.decision == "approve"
    assert granted.edited_args == {"category": "movies"}
    assert granted.reason is None

    denied = session._build_resolution_event(  # pyright: ignore[reportPrivateUsage]
        ctx, turn_id, "c1", "deny", {"category": "movies"}, "wrong account", "u:morten"
    )
    assert denied.decision == "deny"
    assert denied.edited_args is None
    assert denied.reason == "wrong account"

    # The session treats anything that is not exactly "approve" as a denial;
    # the event has to report what happened, not what was asked for.
    shouty = session._build_resolution_event(  # pyright: ignore[reportPrivateUsage]
        ctx, turn_id, "c1", "APPROVE", None, None, "u:morten"
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
        queue, ctx, turn_id, "c1", "approve", {"category": "movies"}, None, "u:morten"
    )

    events = [queue.get_nowait() for _ in range(queue.qsize())]
    assert [type(e).__name__ for e in events] == ["ApprovalGranted", "ApprovalResolved"]
    assert events[1].resolved_by == "u:morten"
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


def _collect(stream) -> list:
    async def _run() -> list:
        return [ev async for ev in stream]

    return asyncio.run(_run())
