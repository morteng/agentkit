"""Tests for AgentSession.resume_with_approval_batch.

Single-decision batches and ApprovalDenied are easy to exercise with
the FakeProvider (one tool_use per assistant message). True multi-pending
batches (one provider message yields N parallel tool_use blocks at once)
need a provider-side feature the FakeProvider doesn't model. The
consumer's adapter-level tests cover the parallel scenario; here we
verify the API contract: shared checkpoint load, per-decision verdict
events, and the unknown-call_id error path.
"""

import pytest

from agentkit import AgentConfig, AgentSession
from agentkit._ids import OwnerId
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


def _make_session(executions: list[dict]) -> tuple[AgentSession, ToolSpec]:
    config = AgentConfig()
    config.guards.approval = RiskBasedApprovalGate()
    config.stores.session = FakeSessionStore()
    config.stores.memory = FakeMemoryStore()
    config.stores.checkpoint = FakeCheckpointStore()

    registry = ToolRegistry()
    registry.register_default_builtins()

    spec = ToolSpec(
        name="ampaera.devices.control",
        description="control device",
        parameters={"type": "object"},
        returns=None,
        risk=RiskLevel.HIGH_WRITE,
        idempotent=False,
        side_effects=SideEffects.EXTERNAL_REVERSIBLE,
        requires_approval=ApprovalPolicy.BY_RISK,
        cache_ttl_seconds=None,
        timeout_seconds=10.0,
    )

    async def handler(args, ctx):
        executions.append(dict(args))
        return ToolResult(
            call_id=ctx.call_id,
            status="ok",
            content=[ContentBlockOut(type="text", text="ok")],
            error=None,
            duration_ms=1,
            cached=False,
        )

    registry.register_builtin(spec, handler)

    provider = FakeProvider().script(
        FakeProvider.tool_call("ampaera.devices.control", {"id": "x"}),
        # After resume, the loop iterates back to streaming; finalize ends turn.
        FakeProvider.tool_call("kit.finalize", {"reason": "done"}),
    )
    session = AgentSession(
        owner=OwnerId("u:1"),
        config=config,
        provider=provider,
        registry=registry,
        model="m",
    )
    return session, spec


@pytest.mark.asyncio
async def test_batch_resume_with_single_approve_executes_tool():
    """One-entry batch: same behaviour as resume_with_approval."""
    executions: list[dict] = []
    session, _ = _make_session(executions)

    needed: ApprovalNeeded | None = None
    async with session.run("control x") as stream:
        async for ev in stream:
            if isinstance(ev, ApprovalNeeded):
                needed = ev
            elif isinstance(ev, TurnEnded):
                assert ev.reason is TurnEndReason.AWAITING_APPROVAL
    assert needed is not None
    assert executions == []

    granted: list[ApprovalGranted] = []
    results: list[ToolCallResult] = []
    async with session.resume_with_approval_batch(
        needed.turn_id,
        [{"call_id": needed.call_id, "decision": "approve"}],
    ) as stream:
        async for ev in stream:
            if isinstance(ev, ApprovalGranted):
                granted.append(ev)
            elif isinstance(ev, ToolCallResult):
                results.append(ev)

    assert len(granted) == 1
    assert granted[0].call_id == needed.call_id
    # The first stream event must be the verdict so consumers can render it
    # before any subsequent tool work — same contract as the sibling
    # resume_with_approval API.
    assert isinstance(granted[0], ApprovalGranted)
    assert executions == [{"id": "x"}]
    assert results and results[0].status == "ok"


@pytest.mark.asyncio
async def test_batch_resume_with_deny_emits_denied_event_and_does_not_execute():
    executions: list[dict] = []
    session, _ = _make_session(executions)

    needed: ApprovalNeeded | None = None
    async with session.run("control x") as stream:
        async for ev in stream:
            if isinstance(ev, ApprovalNeeded):
                needed = ev
    assert needed is not None

    denied: list[ApprovalDenied] = []
    async with session.resume_with_approval_batch(
        needed.turn_id,
        [{"call_id": needed.call_id, "decision": "deny", "reason": "no"}],
    ) as stream:
        async for ev in stream:
            if isinstance(ev, ApprovalDenied):
                denied.append(ev)

    assert len(denied) == 1
    assert denied[0].reason == "no"
    assert executions == []  # tool never ran


@pytest.mark.asyncio
async def test_batch_resume_unknown_call_id_raises_checkpoint_missing():
    """Unknown call_id aborts the resume before the Loop restarts."""
    executions: list[dict] = []
    session, _ = _make_session(executions)

    needed: ApprovalNeeded | None = None
    async with session.run("control x") as stream:
        async for ev in stream:
            if isinstance(ev, ApprovalNeeded):
                needed = ev
    assert needed is not None

    with pytest.raises(CheckpointMissing):
        async with session.resume_with_approval_batch(
            needed.turn_id,
            [{"call_id": "call_does_not_exist", "decision": "approve"}],
        ) as stream:
            async for _ in stream:
                pass
    assert executions == []  # nothing ran
