from typing import Any

import pytest

from agentkit.errors import ToolError
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


@pytest.mark.asyncio
async def test_registry_invokes_registered_builtin():
    reg = ToolRegistry()

    async def handler(args, ctx):
        return ToolResult(
            call_id=ctx.call_id,
            status="ok",
            content=[ContentBlockOut(type="text", text=f"hi {args['n']}")],
            error=None,
            duration_ms=1,
            cached=False,
        )

    reg.register_builtin(_spec("kit.test"), handler)
    specs = reg.list_specs()
    assert any(s.name == "kit.test" for s in specs)

    res = await reg.invoke(ToolCall(id="c1", name="kit.test", arguments={"n": "x"}), ctx=_FakeCtx())  # type: ignore[arg-type]
    assert res.content[0].text == "hi x"


@pytest.mark.asyncio
async def test_registry_rejects_duplicate_registration():
    reg = ToolRegistry()

    async def h(args, ctx): ...

    reg.register_builtin(_spec("kit.test"), h)  # type: ignore[arg-type]
    with pytest.raises(ToolError):
        reg.register_builtin(_spec("kit.test"), h)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_registry_unknown_tool_raises():
    reg = ToolRegistry()
    with pytest.raises(ToolError):
        await reg.invoke(ToolCall(id="c", name="missing", arguments={}), ctx=_FakeCtx())  # type: ignore[arg-type]


class _FakeCtx:
    """Minimal stand-in for TurnContext (defined later)."""

    call_id = "c1"


@pytest.mark.asyncio
async def test_registry_invoke_passes_progress_callback_to_mcp_client() -> None:
    """ToolRegistry.invoke must wire ctx.report_tool_progress to the MCP
    client's on_progress hook so server-side notifications surface as
    user-facing ToolCallProgress events."""
    from datetime import UTC, datetime

    from agentkit.events import ToolCallProgress
    from agentkit.loop.context import FixedClock, TurnContext

    captured_callback: list[Any] = []

    class _FakeMCPClient:
        name = "srv"

        async def initialize(self) -> None: ...
        async def list_tools(self) -> list[ToolSpec]:
            return [_spec("ping")]

        async def call_tool(self, name: str, arguments, *, on_progress=None):
            captured_callback.append(on_progress)
            # Server fires a couple of progress notifications mid-call.
            if on_progress is not None:
                await on_progress("starting", 0.0, 2.0)
                await on_progress("done step 1", 1.0, 2.0)
            return ToolResult(
                call_id="",
                status="ok",
                content=[ContentBlockOut(type="text", text="pong")],
                error=None,
                duration_ms=0,
                cached=False,
            )

        async def shutdown(self) -> None: ...
        async def health_check(self) -> bool:
            return True

    reg = ToolRegistry()
    reg.register_mcp_server("srv", _FakeMCPClient())  # type: ignore[arg-type]
    await reg.initialize_mcp_servers()

    import asyncio

    queue: asyncio.Queue[Any] = asyncio.Queue()
    ctx = TurnContext.empty(clock=FixedClock(datetime.now(UTC)), call_id="call-42")
    ctx.event_queue = queue

    res = await reg.invoke(
        ToolCall(id="call-42", name="srv.ping", arguments={}),
        ctx=ctx,  # type: ignore[arg-type]
    )
    assert res.status == "ok"
    # The callback was wired in.
    assert len(captured_callback) == 1 and captured_callback[0] is not None

    progress_events: list[ToolCallProgress] = []
    while not queue.empty():
        evt = queue.get_nowait()
        assert isinstance(evt, ToolCallProgress)
        progress_events.append(evt)

    assert len(progress_events) == 2
    assert [e.message for e in progress_events] == ["starting", "done step 1"]
    assert [e.progress for e in progress_events] == [0.0, 1.0]
    assert all(e.total == 2.0 for e in progress_events)
    assert all(e.call_id == "call-42" for e in progress_events)
