import pytest

from agentkit import AgentConfig, AgentSession
from agentkit._ids import OwnerId
from agentkit.events import ApprovalNeeded, ToolCallResult, TurnEnded, TurnEndReason
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
