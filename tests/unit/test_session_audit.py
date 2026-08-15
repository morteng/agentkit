"""Approval verdicts reach the audit sink, correctly attributed.

The tool-call record says the vault was deleted. Only this one says who agreed
to it — and whether anyone did at all, which is the case the expiry path
covers.
"""

import asyncio

import pytest

from agentkit import AgentConfig, AgentSession
from agentkit._ids import OwnerId, TurnId, new_id
from agentkit.audit import (
    ACTION_APPROVAL_RESOLVED,
    SOURCE_AGENT,
    SOURCE_HUMAN,
    AuditRecord,
    AuditSink,
    NullAuditSink,
)
from agentkit.errors import ApprovalTimeout
from agentkit.loop.context import TurnContext
from agentkit.providers.fakes import FakeProvider
from agentkit.store.fakes import FakeCheckpointStore, FakeMemoryStore, FakeSessionStore
from agentkit.tools.registry import ToolRegistry


class _Recorder(AuditSink):
    def __init__(self) -> None:
        self.records: list[AuditRecord] = []

    async def record(self, record: AuditRecord) -> None:
        self.records.append(record)


def _session(audit: AuditSink | None = None, *, on_config: bool = False) -> AgentSession:
    config = AgentConfig()
    config.stores.session = FakeSessionStore()
    config.stores.memory = FakeMemoryStore()
    config.stores.checkpoint = FakeCheckpointStore()
    if on_config:
        config.audit_sink = audit
    return AgentSession(
        owner=OwnerId("acme:alice"),
        config=config,
        provider=FakeProvider().script(FakeProvider.text("hi")),
        registry=ToolRegistry(),
        model="m",
        audit_sink=None if on_config else audit,
    )


def _ctx_with_verdict(call_id: str, tool_name: str, *, approved: bool) -> TurnContext:
    ctx = TurnContext.empty()
    bucket = "approved_tool_calls" if approved else "denied_tool_calls"
    ctx.metadata[bucket] = [{"id": call_id, "name": tool_name, "arguments": {}}]
    return ctx


def test_the_default_sink_is_a_no_op_not_none():
    """Every call site emits unconditionally, so the attribute is never None."""
    assert isinstance(_session().audit_sink, NullAuditSink)


def test_the_sink_can_come_from_the_config():
    audit = _Recorder()
    assert _session(audit, on_config=True).audit_sink is audit


def test_the_constructor_wins_over_the_config():
    ctor, configured = _Recorder(), _Recorder()
    config = AgentConfig()
    config.audit_sink = configured
    session = AgentSession(
        owner=OwnerId("acme:alice"),
        config=config,
        provider=FakeProvider().script(FakeProvider.text("hi")),
        registry=ToolRegistry(),
        model="m",
        audit_sink=ctor,
    )
    assert session.audit_sink is ctor


@pytest.mark.asyncio
async def test_a_human_verdict_is_filed_as_human_activity():
    audit = _Recorder()
    session = _session(audit)
    ctx = _ctx_with_verdict("c1", "vault_user_delete", approved=True)
    turn_id = new_id(TurnId)

    await session._emit_verdict(  # pyright: ignore[reportPrivateUsage]
        asyncio.Queue(),
        ctx,
        turn_id,
        "c1",
        "approve",
        {"email": "bob@example.test", "token": "secret"},
        None,
        "acme:alice",
    )

    assert len(audit.records) == 1
    record = audit.records[0]
    assert record.action == ACTION_APPROVAL_RESOLVED
    assert record.actor == "acme:alice"
    # A human's consent is the one row that must never look like agent activity.
    assert record.source == SOURCE_HUMAN
    assert record.target == "vault_user_delete"
    assert record.call_id == "c1"
    assert record.turn_id == str(turn_id)
    assert record.detail["decision"] == "approve"
    assert record.detail["expired"] is False
    # The edit the human made is part of the consent, and is redacted like any
    # other argument projection.
    assert record.detail["edited_args"] == {"email": "bob@example.test", "token": "***"}


@pytest.mark.asyncio
async def test_a_denial_records_the_reason_and_no_arguments():
    audit = _Recorder()
    session = _session(audit)
    ctx = _ctx_with_verdict("c1", "torrent_add", approved=False)

    await session._emit_verdict(  # pyright: ignore[reportPrivateUsage]
        asyncio.Queue(),
        ctx,
        new_id(TurnId),
        "c1",
        "deny",
        {"url": "magnet:?xt=x"},
        "wrong account",
        "acme:alice",
    )

    detail = audit.records[0].detail
    assert detail["decision"] == "deny"
    assert detail["reason"] == "wrong account"
    assert "edited_args" not in detail


@pytest.mark.asyncio
async def test_a_shouted_approval_is_audited_as_the_denial_it_actually_is():
    """The session denies anything that is not exactly "approve"; the record has
    to report what happened, not what was asked for."""
    audit = _Recorder()
    session = _session(audit)
    ctx = _ctx_with_verdict("c1", "torrent_add", approved=False)

    await session._emit_verdict(  # pyright: ignore[reportPrivateUsage]
        asyncio.Queue(), ctx, new_id(TurnId), "c1", "APPROVE", {"a": 1}, None, "acme:alice"
    )

    assert audit.records[0].detail["decision"] == "deny"
    assert "edited_args" not in audit.records[0].detail


@pytest.mark.asyncio
async def test_a_verdict_the_runtime_reached_is_agent_activity():
    audit = _Recorder()
    session = _session(audit)
    ctx = _ctx_with_verdict("c2", "torrent_add", approved=False)

    await session._emit_verdict(  # pyright: ignore[reportPrivateUsage]
        asyncio.Queue(),
        ctx,
        new_id(TurnId),
        "c2",
        "deny",
        None,
        "not resolved in this resume",
        "system",
    )

    assert audit.records[0].source == SOURCE_AGENT
    assert audit.records[0].actor == "system"


@pytest.mark.asyncio
async def test_an_expired_approval_is_recorded_even_if_nobody_reads_the_stream():
    """Silence is not consent, and an unanswered approval that leaves no trace
    is indistinguishable from one that was never asked for."""
    audit = _Recorder()
    session = _session(audit)
    turn_id = new_id(TurnId)
    exc = ApprovalTimeout("approval window expired", call_ids=["c1", "c2"])

    stream = session._approval_timeout_stream(turn_id, exc)  # pyright: ignore[reportPrivateUsage]
    # Pull one event only: the records must already all be written.
    await anext(stream)

    assert [r.call_id for r in audit.records] == ["c1", "c2"]
    for record in audit.records:
        assert record.detail["expired"] is True
        assert record.detail["decision"] == "deny"
        assert record.source == SOURCE_AGENT
        assert record.actor == "system"
        # The checkpoint is gone by now, so the tool name is genuinely unknown
        # rather than guessed.
        assert record.target == ""


@pytest.mark.asyncio
async def test_a_broken_sink_does_not_break_the_resume():
    class _Exploding(AuditSink):
        async def record(self, record: AuditRecord) -> None:
            raise RuntimeError("ledger unavailable")

    session = _session(_Exploding())
    queue: asyncio.Queue = asyncio.Queue()
    ctx = _ctx_with_verdict("c1", "torrent_add", approved=True)

    await session._emit_verdict(  # pyright: ignore[reportPrivateUsage]
        queue, ctx, new_id(TurnId), "c1", "approve", None, None, "acme:alice"
    )

    # The verdict events still reached the consumer.
    assert queue.qsize() == 2


@pytest.mark.asyncio
async def test_the_dispatcher_is_built_with_the_session_sink():
    """Wiring check: the sink on the session is the sink tool calls use."""
    audit = _Recorder()
    session = _session(audit)
    deps = session._build_deps()  # pyright: ignore[reportPrivateUsage]
    assert deps["audit_sink"] is audit
    assert deps["dispatcher"]._audit is audit  # pyright: ignore[reportPrivateUsage]
