import pytest

from agentkit.guards.approval import RiskBasedApprovalGate
from agentkit.loop.context import TurnContext
from agentkit.loop.handlers.tool_executing import handle_tool_executing
from agentkit.loop.handlers.tool_phase import handle_tool_phase
from agentkit.loop.handlers.tool_results import handle_tool_results
from agentkit.loop.phase import Phase
from agentkit.loop.tool_dispatcher import DispatchPolicy, ToolDispatcher
from agentkit.tools.registry import ToolRegistry
from agentkit.tools.spec import (
    ApprovalPolicy,
    ContentBlockOut,
    RiskLevel,
    SideEffects,
    ToolResult,
    ToolSpec,
)


def _spec(name: str, risk: RiskLevel = RiskLevel.READ) -> ToolSpec:
    return ToolSpec(
        name=name,
        description="d",
        parameters={"type": "object"},
        returns=None,
        risk=risk,
        idempotent=True,
        side_effects=SideEffects.NONE,
        requires_approval=ApprovalPolicy.BY_RISK,
        cache_ttl_seconds=None,
        timeout_seconds=10.0,
    )


def _make_deps(registry: ToolRegistry, *, with_writes: bool = False):
    return {
        "registry": registry,
        "approval_gate": RiskBasedApprovalGate(),
        "dispatcher": ToolDispatcher(registry=registry, policy=DispatchPolicy()),
    }


@pytest.mark.asyncio
async def test_tool_phase_auto_approve_routes_to_executing():
    reg = ToolRegistry()

    async def h(args, ctx):
        return ToolResult(
            call_id=ctx.call_id,
            status="ok",
            content=[ContentBlockOut(type="text", text="ok")],
            error=None,
            duration_ms=1,
            cached=False,
        )

    reg.register_builtin(_spec("kit.read"), h)

    ctx = TurnContext.empty()
    ctx.metadata["pending_tool_calls"] = [{"id": "c1", "name": "kit.read", "arguments": {}}]
    next_ = await handle_tool_phase(ctx, _make_deps(reg))
    assert next_ is Phase.TOOL_EXECUTING


@pytest.mark.asyncio
async def test_tool_phase_high_write_routes_to_approval_wait():
    reg = ToolRegistry()

    async def h(args, ctx):
        return ToolResult(
            call_id=ctx.call_id, status="ok", content=[], error=None, duration_ms=1, cached=False
        )

    reg.register_builtin(_spec("kit.write", risk=RiskLevel.HIGH_WRITE), h)

    ctx = TurnContext.empty()
    ctx.metadata["pending_tool_calls"] = [{"id": "c1", "name": "kit.write", "arguments": {}}]
    next_ = await handle_tool_phase(ctx, _make_deps(reg))
    assert next_ is Phase.APPROVAL_WAIT


@pytest.mark.asyncio
async def test_tool_executing_runs_dispatched_calls():
    reg = ToolRegistry()

    async def h(args, ctx):
        return ToolResult(
            call_id=ctx.call_id,
            status="ok",
            content=[ContentBlockOut(type="text", text="result")],
            error=None,
            duration_ms=1,
            cached=False,
        )

    reg.register_builtin(_spec("kit.read"), h)

    ctx = TurnContext.empty()
    ctx.metadata["approved_tool_calls"] = [{"id": "c1", "name": "kit.read", "arguments": {}}]
    ctx.metadata["denied_tool_calls"] = []
    deps = _make_deps(reg)
    next_ = await handle_tool_executing(ctx, deps)
    assert next_ is Phase.TOOL_RESULTS
    assert len(ctx.metadata["tool_results"]) == 1


@pytest.mark.asyncio
async def test_tool_results_routes_to_finalize_check_when_finalize_was_called():
    ctx = TurnContext.empty()
    ctx.metadata["tool_results"] = []
    ctx.finalize_called = True
    next_ = await handle_tool_results(ctx, {})
    assert next_ is Phase.FINALIZE_CHECK


@pytest.mark.asyncio
async def test_tool_results_routes_to_context_build_when_more_iteration_needed():
    ctx = TurnContext.empty()
    ctx.metadata["tool_results"] = []
    next_ = await handle_tool_results(ctx, {})
    assert next_ is Phase.CONTEXT_BUILD
