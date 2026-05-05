import asyncio

import pytest

from agentkit import AgentConfig, AgentSession
from agentkit._ids import OwnerId
from agentkit.errors import ApprovalTimeout
from agentkit.events import (
    ApprovalDenied,
    ApprovalGranted,
    ApprovalNeeded,
    ErrorCode,
    Errored,
    ToolCallResult,
    TurnEnded,
    TurnEndReason,
)
from agentkit.guards.approval import RiskBasedApprovalGate
from agentkit.providers.fakes import FakeProvider
from agentkit.store.fakes import FakeCheckpointStore, FakeMemoryStore, FakeSessionStore
from agentkit.tools.builtin import DEFAULT_BUILTINS
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


@pytest.mark.asyncio
async def test_approval_suspend_then_resume_executes_tool():
    """Provider returns a HIGH_WRITE call; first turn suspends; resume executes."""
    config = AgentConfig()
    config.guards.approval = RiskBasedApprovalGate()
    config.stores.session = FakeSessionStore()
    config.stores.memory = FakeMemoryStore()
    config.stores.checkpoint = FakeCheckpointStore()

    registry = ToolRegistry()
    for spec, handler in DEFAULT_BUILTINS:
        registry.register_builtin(spec, handler)

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
    executions: list[dict] = []

    async def control(args, ctx):
        executions.append(args)
        return ToolResult(
            call_id=ctx.call_id,
            status="ok",
            content=[ContentBlockOut(type="text", text="off")],
            error=None,
            duration_ms=1,
            cached=False,
        )

    registry.register_builtin(spec, control)

    provider = FakeProvider().script(
        FakeProvider.tool_call("ampaera.devices.control", {"id": "heat-pump"}),
        # After resume executes the tool, the loop iterates back to streaming.
        # Have the provider finalize so the resumed turn ends COMPLETED.
        FakeProvider.tool_call("kit.finalize", {"reason": "done"}),
    )
    session = AgentSession(
        owner=OwnerId("u:1"),
        config=config,
        provider=provider,
        registry=registry,
        model="m",
    )

    # First turn: should suspend on approval.
    needed: ApprovalNeeded | None = None
    async with session.run("turn off the heat pump") as stream:
        async for ev in stream:
            if isinstance(ev, ApprovalNeeded):
                needed = ev
            elif isinstance(ev, TurnEnded):
                assert ev.reason is TurnEndReason.AWAITING_APPROVAL
    assert needed is not None
    assert executions == []  # tool not yet called

    # Second turn: resume with approval.
    results: list[ToolCallResult] = []
    async with session.resume_with_approval(
        needed.turn_id,
        needed.call_id,
        decision="approve",
    ) as stream:
        async for ev in stream:
            if isinstance(ev, ToolCallResult):
                results.append(ev)
            elif isinstance(ev, TurnEnded):
                assert ev.reason is TurnEndReason.COMPLETED
    assert executions and executions[0] == {"id": "heat-pump"}
    assert results and results[0].status == "ok"


def _make_approval_session() -> tuple[AgentSession, ToolSpec, list[dict]]:
    """Helper: session + HIGH_WRITE tool that records executions."""
    config = AgentConfig()
    config.guards.approval = RiskBasedApprovalGate()
    config.stores.session = FakeSessionStore()
    config.stores.memory = FakeMemoryStore()
    config.stores.checkpoint = FakeCheckpointStore()

    registry = ToolRegistry()
    registry.register_default_builtins()

    spec = ToolSpec(
        name="ampaera.devices.delete",
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
    executions: list[dict] = []

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

    provider = FakeProvider().script(
        FakeProvider.tool_call("ampaera.devices.delete", {"id": "x"}),
        FakeProvider.tool_call("kit.finalize", {"reason": "done"}),
    )
    session = AgentSession(
        owner=OwnerId("u:1"),
        config=config,
        provider=provider,
        registry=registry,
        model="m",
    )
    return session, spec, executions


@pytest.mark.asyncio
async def test_resume_approve_emits_approval_granted_first():
    """F14: ApprovalGranted is yielded as the first event of the resumed stream."""
    session, _, _ = _make_approval_session()

    needed: ApprovalNeeded | None = None
    async with session.run("delete x") as stream:
        async for ev in stream:
            if isinstance(ev, ApprovalNeeded):
                needed = ev
    assert needed is not None

    events: list = []
    async with session.resume_with_approval(
        needed.turn_id, needed.call_id, decision="approve"
    ) as s:
        async for ev in s:
            events.append(ev)
    granted = next((e for e in events if isinstance(e, ApprovalGranted)), None)
    assert granted is not None
    assert granted.call_id == needed.call_id
    assert granted.edited_args is None
    # Granted must arrive before the eventual ToolCallResult.
    granted_idx = events.index(granted)
    tcr_idx = next(i for i, e in enumerate(events) if isinstance(e, ToolCallResult))
    assert granted_idx < tcr_idx


@pytest.mark.asyncio
async def test_resume_approve_with_edited_args_surfaces_in_event():
    """F14: ApprovalGranted.edited_args reflects the override the user passed."""
    session, _, executions = _make_approval_session()

    needed: ApprovalNeeded | None = None
    async with session.run("delete x") as stream:
        async for ev in stream:
            if isinstance(ev, ApprovalNeeded):
                needed = ev
    assert needed is not None

    granted: ApprovalGranted | None = None
    async with session.resume_with_approval(
        needed.turn_id,
        needed.call_id,
        decision="approve",
        edited_args={"id": "y_override"},
    ) as s:
        async for ev in s:
            if isinstance(ev, ApprovalGranted):
                granted = ev
    assert granted is not None
    assert granted.edited_args == {"id": "y_override"}
    # The handler actually saw the override.
    assert executions == [{"id": "y_override"}]


@pytest.mark.asyncio
async def test_resume_deny_emits_approval_denied_first():
    """F14: ApprovalDenied carries the reason and arrives before tool_call_result."""
    session, _, executions = _make_approval_session()

    needed: ApprovalNeeded | None = None
    async with session.run("delete x") as stream:
        async for ev in stream:
            if isinstance(ev, ApprovalNeeded):
                needed = ev
    assert needed is not None

    events: list = []
    async with session.resume_with_approval(
        needed.turn_id,
        needed.call_id,
        decision="deny",
        reason="not authorized",
    ) as s:
        async for ev in s:
            events.append(ev)
    denied = next((e for e in events if isinstance(e, ApprovalDenied)), None)
    assert denied is not None
    assert denied.reason == "not authorized"
    assert executions == []  # nothing was actually executed


@pytest.mark.asyncio
async def test_resume_after_timeout_emits_errored_and_does_not_execute_tool():
    """F24: a stale approval surfaces as Errored(APPROVAL_TIMEOUT) and never runs the tool."""
    session, _, executions = _make_approval_session()
    session.config.guards.approval_timeout_seconds = 0.05  # 50ms

    needed: ApprovalNeeded | None = None
    async with session.run("delete x") as stream:
        async for ev in stream:
            if isinstance(ev, ApprovalNeeded):
                needed = ev
    assert needed is not None

    # Sleep past the timeout.
    await asyncio.sleep(0.2)

    events: list = []
    async with session.resume_with_approval(
        needed.turn_id, needed.call_id, decision="approve"
    ) as s:
        async for ev in s:
            events.append(ev)

    errored = next((e for e in events if isinstance(e, Errored)), None)
    assert errored is not None
    assert errored.code == ErrorCode.APPROVAL_TIMEOUT
    assert any(isinstance(e, TurnEnded) and e.reason is TurnEndReason.ERROR for e in events)
    assert executions == []  # tool MUST NOT have run


@pytest.mark.asyncio
async def test_load_resume_context_raises_approval_timeout_directly():
    """F24: lower-level _load_resume_context still raises ApprovalTimeout for callers
    that prefer the exception path over the event stream."""
    session, _, _ = _make_approval_session()
    session.config.guards.approval_timeout_seconds = 0.05

    needed: ApprovalNeeded | None = None
    async with session.run("delete x") as stream:
        async for ev in stream:
            if isinstance(ev, ApprovalNeeded):
                needed = ev
    assert needed is not None

    await asyncio.sleep(0.2)

    with pytest.raises(ApprovalTimeout):
        await session._load_resume_context(needed.turn_id)


@pytest.mark.asyncio
async def test_default_builtins_no_longer_includes_request_approval():
    """F15: kit.request_approval is exported but not enabled by default."""
    registry = ToolRegistry()
    registry.register_default_builtins()
    names = {s.name for s in registry.list_specs()}
    assert "kit.request_approval" not in names
    assert "kit.subagent.spawn" in names  # but subagent stays
    assert "kit.finalize" in names
