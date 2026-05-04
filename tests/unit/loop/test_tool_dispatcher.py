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
    reg = ToolRegistry()
    started = []

    async def slow_handler(name):
        async def h(args, ctx):
            started.append(name)
            await asyncio.sleep(0.05)
            return _ok(ctx.call_id, name)

        return h

    reg.register_builtin(_spec("kit.a"), await slow_handler("a"))
    reg.register_builtin(_spec("kit.b"), await slow_handler("b"))

    disp = ToolDispatcher(registry=reg, policy=DispatchPolicy(max_parallel=8))

    calls = [
        ToolCall(id="c1", name="kit.a", arguments={}),
        ToolCall(id="c2", name="kit.b", arguments={}),
    ]
    import time

    t0 = time.perf_counter()
    results = await disp.run(calls, ctx=_FakeCtx())
    elapsed = time.perf_counter() - t0

    assert len(results) == 2
    # Sequential would be ~0.10s; parallel should be < 0.08s.
    assert elapsed < 0.08


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


class _FakeCtx:
    call_id = "c1"
