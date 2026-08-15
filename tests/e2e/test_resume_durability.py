"""Approval resume, from the store's point of view.

``resume_with_approval`` used to stream a perfectly good turn and persist none
of it: the tool results and the final assistant reply existed only in the
consumer's process. These tests assert the STORE CONTENT after a resume, plus
the two other ways a resume could quietly lose a turn — a bad call_id
destroying the checkpoint, and unresolved pending calls being dropped.
"""

import asyncio

import pytest

from agentkit import AgentConfig, AgentSession
from agentkit._content import ToolResultBlock, ToolUseBlock
from agentkit._ids import CheckpointId, OwnerId
from agentkit._messages import Message, MessageRole
from agentkit.errors import CheckpointMissing
from agentkit.events import (
    ApprovalDenied,
    ApprovalGranted,
    ApprovalNeeded,
    ToolCallResult,
    TurnEnded,
    TurnEndReason,
)
from agentkit.guards.approval import RiskBasedApprovalGate
from agentkit.providers.fakes import FakeProvider
from agentkit.session import SYNTHETIC_TOOL_RESULT_ANNOTATION
from agentkit.store.fakes import FakeCheckpointStore, FakeMemoryStore, FakeSessionStore
from agentkit.tools.registry import ToolRegistry
from agentkit.tools.spec import (
    ApprovalPolicy,
    ContentBlockOut,
    RiskLevel,
    SideEffects,
    ToolResult,
    ToolSpec,
)

pytestmark = pytest.mark.e2e

_FINALIZE = FakeProvider.tool_call(
    "kit.finalize",
    {
        "status": "done",
        "intent_kind": "action",
        "summary": "Deleted the device.",
        "actions_performed": [{"tool": "devices.delete", "description": "deleted device"}],
    },
)


def _dangling_tool_use_ids(messages: list[Message]) -> list[str]:
    used = [b.id for m in messages for b in m.content if isinstance(b, ToolUseBlock)]
    answered = {
        b.tool_use_id for m in messages for b in m.content if isinstance(b, ToolResultBlock)
    }
    return [call_id for call_id in used if call_id not in answered]


def _answered_tool_use_ids(messages: list[Message]) -> set[str]:
    return {b.tool_use_id for m in messages for b in m.content if isinstance(b, ToolResultBlock)}


def _make_session(*responses, executions: list[dict]) -> AgentSession:
    """Session with one HIGH_WRITE tool (so the first turn suspends) that records
    every execution."""
    config = AgentConfig()
    config.guards.approval = RiskBasedApprovalGate()
    config.stores.session = FakeSessionStore()
    config.stores.memory = FakeMemoryStore()
    config.stores.checkpoint = FakeCheckpointStore()

    registry = ToolRegistry()
    registry.register_default_builtins()
    spec = ToolSpec(
        name="REDACTED.devices.delete",
        description="delete device (irreversible)",
        parameters={"type": "object"},
        returns=None,
        risk=RiskLevel.HIGH_WRITE,
        idempotent=False,
        side_effects=SideEffects.EXTERNAL_IRREVERSIBLE,
        requires_approval=ApprovalPolicy.BY_RISK,
        cache_ttl_seconds=None,
        timeout_seconds=10.0,
    )

    async def handler(args, ctx):
        executions.append(dict(args))
        return ToolResult(
            call_id=ctx.call_id,
            status="ok",
            content=[ContentBlockOut(type="text", text="deleted")],
            error=None,
            duration_ms=1,
            cached=False,
        )

    registry.register_builtin(spec, handler)

    return AgentSession(
        owner=OwnerId("u:1"),
        config=config,
        provider=FakeProvider().script(*responses),
        registry=registry,
        model="m",
    )


async def _suspend(session: AgentSession, text: str) -> list[ApprovalNeeded]:
    needed: list[ApprovalNeeded] = []
    async with session.run(text) as stream:
        async for ev in stream:
            if isinstance(ev, ApprovalNeeded):
                needed.append(ev)
            elif isinstance(ev, TurnEnded):
                assert ev.reason is TurnEndReason.AWAITING_APPROVAL
    assert needed
    return needed


async def _stored(session: AgentSession) -> list[Message]:
    return await session.config.stores.session.list_messages(session.id, limit=1000)


# --- F: the resumed turn was never persisted --------------------------------


@pytest.mark.asyncio
async def test_resume_persists_the_tool_result_and_the_rest_of_the_turn():
    executions: list[dict] = []
    session = _make_session(
        FakeProvider.tool_call("REDACTED.devices.delete", {"id": "x"}),
        _FINALIZE,
        executions=executions,
    )
    needed = await _suspend(session, "delete x")

    before = await _stored(session)
    # The suspended turn is on record with its call still open — that is the
    # checkpoint's job to resolve, so nothing may be synthesized for it yet.
    assert _dangling_tool_use_ids(before) == [needed[0].call_id]
    assert not [m for m in before if m.metadata.annotations.get(SYNTHETIC_TOOL_RESULT_ANNOTATION)]

    async with session.resume_with_approval(
        needed[0].turn_id, needed[0].call_id, decision="approve"
    ) as stream:
        async for _ev in stream:
            pass

    after = await _stored(session)
    assert len(after) > len(before), "the resumed turn persisted nothing"
    # Every message the suspended turn wrote is still there, untouched...
    assert [m.id for m in after[: len(before)]] == [m.id for m in before]
    # ...and the approved call now has its real result in the store.
    assert needed[0].call_id in _answered_tool_use_ids(after)
    assert _dangling_tool_use_ids(after) == []
    assert executions == [{"id": "x"}]
    assert len({m.id for m in after}) == len(after)


@pytest.mark.asyncio
async def test_turn_after_a_resume_loads_a_well_formed_transcript():
    executions: list[dict] = []
    session = _make_session(
        FakeProvider.tool_call("REDACTED.devices.delete", {"id": "x"}),
        _FINALIZE,
        FakeProvider.text("anything else?"),
        executions=executions,
    )
    needed = await _suspend(session, "delete x")
    async with session.resume_with_approval(
        needed[0].turn_id, needed[0].call_id, decision="approve"
    ) as stream:
        async for _ev in stream:
            pass

    async with session.run("thanks") as stream:
        async for _ev in stream:
            pass

    assert _dangling_tool_use_ids(await _stored(session)) == []


@pytest.mark.asyncio
async def test_denied_resume_persists_the_denial_result():
    """A denied call still needs a result in the store, or the history is broken."""
    executions: list[dict] = []
    session = _make_session(
        FakeProvider.tool_call("REDACTED.devices.delete", {"id": "x"}),
        _FINALIZE,
        executions=executions,
    )
    needed = await _suspend(session, "delete x")

    async with session.resume_with_approval(
        needed[0].turn_id, needed[0].call_id, decision="deny", reason="no"
    ) as stream:
        async for _ev in stream:
            pass

    assert executions == []
    assert _dangling_tool_use_ids(await _stored(session)) == []


# --- F: a bad call_id destroyed the checkpoint ------------------------------


@pytest.mark.asyncio
async def test_unknown_call_id_leaves_the_checkpoint_resumable():
    executions: list[dict] = []
    session = _make_session(
        FakeProvider.tool_call("REDACTED.devices.delete", {"id": "x"}),
        _FINALIZE,
        executions=executions,
    )
    needed = await _suspend(session, "delete x")
    ckpt_id = CheckpointId(f"approval:{needed[0].turn_id}")

    with pytest.raises(CheckpointMissing):
        async with session.resume_with_approval(
            needed[0].turn_id, "not-a-real-call-id", decision="approve"
        ) as stream:
            async for _ev in stream:
                pass

    # The checkpoint SURVIVES the bad verdict...
    assert await session.config.stores.checkpoint.load(ckpt_id) is not None
    assert executions == []

    # ...so the user can correct the call_id and the turn still runs.
    async with session.resume_with_approval(
        needed[0].turn_id, needed[0].call_id, decision="approve"
    ) as stream:
        async for _ev in stream:
            pass
    assert executions == [{"id": "x"}]
    assert _dangling_tool_use_ids(await _stored(session)) == []


@pytest.mark.asyncio
async def test_duplicate_call_id_in_a_batch_is_rejected_before_anything_runs():
    executions: list[dict] = []
    session = _make_session(
        FakeProvider.tool_call("REDACTED.devices.delete", {"id": "x"}),
        _FINALIZE,
        executions=executions,
    )
    needed = await _suspend(session, "delete x")
    ckpt_id = CheckpointId(f"approval:{needed[0].turn_id}")
    decisions = [
        {"call_id": needed[0].call_id, "decision": "approve"},
        {"call_id": needed[0].call_id, "decision": "deny"},
    ]

    with pytest.raises(CheckpointMissing):
        async with session.resume_with_approval_batch(needed[0].turn_id, decisions) as stream:
            async for _ev in stream:
                pass

    assert await session.config.stores.checkpoint.load(ckpt_id) is not None
    assert executions == []


# --- F: a single-call resume dropped the other pending calls ----------------


@pytest.mark.asyncio
async def test_single_call_resume_auto_denies_the_calls_it_did_not_rule_on():
    """The second pending call must not vanish: it is denied, visibly, with a
    reason, and it gets a result in the transcript like any other call."""
    executions: list[dict] = []
    session = _make_session(
        FakeProvider.tool_calls(
            [
                ("REDACTED.devices.delete", {"id": "a"}),
                ("REDACTED.devices.delete", {"id": "b"}),
            ]
        ),
        _FINALIZE,
        executions=executions,
    )
    needed = await _suspend(session, "delete a and b")
    assert len(needed) == 2

    events: list = []
    async with session.resume_with_approval(
        needed[0].turn_id, needed[0].call_id, decision="approve"
    ) as stream:
        async for ev in stream:
            events.append(ev)

    granted = [e for e in events if isinstance(e, ApprovalGranted)]
    denied = [e for e in events if isinstance(e, ApprovalDenied)]
    assert [g.call_id for g in granted] == [needed[0].call_id]
    assert [d.call_id for d in denied] == [needed[1].call_id]
    assert denied[0].reason == "not resolved in this resume"

    # Only the approved call ran, and BOTH calls have a result on the record.
    assert executions == [{"id": "a"}]
    results = {e.call_id: e.status for e in events if isinstance(e, ToolCallResult)}
    assert results[needed[0].call_id] == "ok"
    assert results[needed[1].call_id] == "denied"
    assert _dangling_tool_use_ids(await _stored(session)) == []


# --- F: an expired approval left the transcript permanently broken ----------


@pytest.mark.asyncio
async def test_expired_approval_closes_the_open_call_in_the_store():
    """The checkpoint is gone and the tool will never run — so the transcript
    must not keep a tool_use that nothing can ever answer."""
    executions: list[dict] = []
    session = _make_session(
        FakeProvider.tool_call("REDACTED.devices.delete", {"id": "x"}),
        _FINALIZE,
        executions=executions,
    )
    session.config.guards.approval_timeout_seconds = 0.05
    needed = await _suspend(session, "delete x")
    assert _dangling_tool_use_ids(await _stored(session)) == [needed[0].call_id]

    await asyncio.sleep(0.2)

    async with session.resume_with_approval(
        needed[0].turn_id, needed[0].call_id, decision="approve"
    ) as stream:
        async for _ev in stream:
            pass

    assert executions == []
    stored = await _stored(session)
    assert _dangling_tool_use_ids(stored) == []
    closer = next(m for m in stored if m.metadata.annotations.get(SYNTHETIC_TOOL_RESULT_ANNOTATION))
    assert closer.role is MessageRole.TOOL
