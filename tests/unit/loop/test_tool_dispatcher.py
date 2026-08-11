import asyncio

import pytest

from agentkit.loop.tool_dispatcher import DispatchPolicy, ToolDispatcher
from agentkit.tools.registry import ToolRegistry
from agentkit.tools.spec import (
    ApprovalPolicy,
    ContentBlockOut,
    RiskLevel,
    SideEffects,
    ToolCall,
    ToolResult,
    ToolSpec,
)


def _spec(name: str, risk: RiskLevel = RiskLevel.READ, idem: bool = True) -> ToolSpec:
    return ToolSpec(
        name=name,
        description="d",
        parameters={"type": "object"},
        returns=None,
        risk=risk,
        idempotent=idem,
        side_effects=SideEffects.NONE,
        requires_approval=ApprovalPolicy.NEVER,
        cache_ttl_seconds=None,
        timeout_seconds=10.0,
    )


def _ok(call_id: str, text: str) -> ToolResult:
    return ToolResult(
        call_id=call_id,
        status="ok",
        content=[ContentBlockOut(type="text", text=text)],
        error=None,
        duration_ms=1,
        cached=False,
    )


@pytest.mark.asyncio
async def test_reads_dispatch_in_parallel():
    """Both reads must be in flight at once — proven by rendezvous, not by a clock.

    This asserted ``elapsed < 0.08`` against two 0.05s handlers, leaving 0.03s
    of headroom over the ideal. That measures the machine as much as the
    dispatcher, and it flaked at 0.096s on a loaded box.

    A barrier turns the same claim into a structural one: neither handler can
    pass ``wait()`` until the other has reached it, so the call returns at all
    ONLY if both were running concurrently. A sequential dispatcher blocks the
    first handler forever, which ``wait_for`` reports as a timeout. The
    generous 5s ceiling is a deadlock detector, not a performance budget — load
    can make this test slow, but it can no longer make it wrong.
    """
    reg = ToolRegistry()
    both_running = asyncio.Barrier(2)

    async def rendezvous_handler(name):
        async def h(args, ctx):
            await both_running.wait()
            return _ok(ctx.call_id, name)

        return h

    reg.register_builtin(_spec("kit.a"), await rendezvous_handler("a"))
    reg.register_builtin(_spec("kit.b"), await rendezvous_handler("b"))

    disp = ToolDispatcher(registry=reg, policy=DispatchPolicy(max_parallel=8))

    calls = [
        ToolCall(id="c1", name="kit.a", arguments={}),
        ToolCall(id="c2", name="kit.b", arguments={}),
    ]

    results = await asyncio.wait_for(disp.run(calls, ctx=_FakeCtx()), timeout=5)

    assert len(results) == 2


@pytest.mark.asyncio
async def test_writes_dispatch_sequentially():
    reg = ToolRegistry()
    order: list[str] = []

    def make(name):
        async def h(args, ctx):
            order.append(f"start-{name}")
            await asyncio.sleep(0.01)
            order.append(f"end-{name}")
            return _ok(ctx.call_id, name)

        return h

    reg.register_builtin(_spec("kit.a", risk=RiskLevel.HIGH_WRITE, idem=False), make("a"))
    reg.register_builtin(_spec("kit.b", risk=RiskLevel.HIGH_WRITE, idem=False), make("b"))

    disp = ToolDispatcher(registry=reg, policy=DispatchPolicy(max_parallel=8))
    calls = [
        ToolCall(id="c1", name="kit.a", arguments={}),
        ToolCall(id="c2", name="kit.b", arguments={}),
    ]
    await disp.run(calls, ctx=_FakeCtx())
    # Strict sequencing: every "start-X" precedes "end-X" before the next "start-Y".
    assert order == ["start-a", "end-a", "start-b", "end-b"]


@pytest.mark.asyncio
async def test_unknown_tool_returns_error_result_not_raises():
    """Defense-in-depth: if an unknown tool name reaches the dispatcher it
    must produce an error ToolResult, never raise. A raised exception bubbles
    to the orchestrator and ends the turn with no result for the model."""
    reg = ToolRegistry()  # empty — 'ghost' is unregistered
    disp = ToolDispatcher(registry=reg, policy=DispatchPolicy(max_parallel=8))
    results = await disp.run([ToolCall(id="c1", name="ghost", arguments={})], ctx=_FakeCtx())
    assert len(results) == 1
    assert results[0].status == "error"
    assert results[0].call_id == "c1"


class _FakeCtx:
    call_id = "c1"
