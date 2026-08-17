import asyncio

import pytest

from agentkit.events import ToolCallResult
from agentkit.events.approval import UNNAMED_TOOL
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


@pytest.mark.asyncio
async def test_tool_results_sets_max_iterations_suspend_reason_when_budget_hit():
    """Hitting the iteration cap must surface as ``MAX_ITERATIONS`` on
    TurnEnded, not the default COMPLETED. The handler signals this by
    setting ``suspend_reason`` in metadata; the orchestrator picks it up.
    """
    from agentkit.events.lifecycle import TurnEndReason

    ctx = TurnContext.empty()
    ctx.metadata["tool_results"] = []
    ctx.metadata["iterations"] = 9  # next iteration hits 10 == max
    next_ = await handle_tool_results(ctx, {"max_iterations": 10})
    assert next_ is Phase.FINALIZE_CHECK
    assert ctx.metadata["max_iterations_hit"] is True
    assert ctx.metadata["suspend_reason"] == TurnEndReason.MAX_ITERATIONS.value


@pytest.mark.asyncio
async def test_tool_results_does_not_overwrite_existing_suspend_reason():
    """If a prior handler already set suspend_reason (e.g. AWAITING_APPROVAL),
    the iteration-budget check must not clobber it.
    """
    from agentkit.events.lifecycle import TurnEndReason

    ctx = TurnContext.empty()
    ctx.metadata["tool_results"] = []
    ctx.metadata["iterations"] = 9
    ctx.metadata["suspend_reason"] = TurnEndReason.AWAITING_APPROVAL.value
    await handle_tool_results(ctx, {"max_iterations": 10})
    assert ctx.metadata["suspend_reason"] == TurnEndReason.AWAITING_APPROVAL.value


@pytest.mark.asyncio
async def test_tool_results_event_carries_error_and_content():
    """F19: ToolCallResult event must propagate ToolError + content for failed calls."""
    import asyncio

    from agentkit.events import ToolCallResult
    from agentkit.tools.spec import ToolError

    ctx = TurnContext.empty()
    ctx.event_queue = asyncio.Queue()
    ctx.metadata["tool_results"] = [
        ToolResult(
            call_id="c1",
            status="error",
            content=[ContentBlockOut(type="text", text="oops")],
            error=ToolError(code="boom", message="something exploded", retryable=True),
            duration_ms=42,
            cached=False,
        )
    ]

    await handle_tool_results(ctx, {})

    ev = ctx.event_queue.get_nowait()
    assert isinstance(ev, ToolCallResult)
    assert ev.status == "error"
    assert ev.error is not None
    assert ev.error.code == "boom"
    assert ev.error.message == "something exploded"
    assert ev.error.retryable is True
    assert len(ev.content) == 1
    assert ev.content[0].text == "oops"


@pytest.mark.asyncio
async def test_tool_results_aborts_after_max_consecutive_errors():
    """F20: 3 back-to-back errors from the same tool transitions to ERRORED."""
    import asyncio

    from agentkit.tools.spec import ToolError

    def _make_ctx_with_one_error_for(name: str):
        c = TurnContext.empty()
        c.event_queue = asyncio.Queue()
        c.metadata["approved_tool_calls"] = [{"id": "c1", "name": name, "arguments": {}}]
        c.metadata["denied_tool_calls"] = []
        c.metadata["tool_results"] = [
            ToolResult(
                call_id="c1",
                status="error",
                content=[],
                error=ToolError(code="boom", message="x"),
                duration_ms=0,
                cached=False,
            )
        ]
        return c

    ctx = _make_ctx_with_one_error_for("kit.broken")
    ctx.metadata["consecutive_tool_errors"] = {"kit.broken": 2}  # this would be the 3rd
    deps = {"max_consecutive_tool_errors": 3}
    next_ = await handle_tool_results(ctx, deps)
    assert next_ is Phase.ERRORED
    assert ctx.metadata["tool_error_loop"]["tool"] == "kit.broken"
    assert ctx.metadata["tool_error_loop"]["count"] == 3
    assert ctx.metadata["tool_error_loop"]["last_error"]["code"] == "boom"


@pytest.mark.asyncio
async def test_tool_results_resets_counter_on_success():
    """F20: a successful call resets the consecutive-error counter for that tool."""
    import asyncio

    ctx = TurnContext.empty()
    ctx.event_queue = asyncio.Queue()
    ctx.metadata["approved_tool_calls"] = [{"id": "c1", "name": "kit.recovered", "arguments": {}}]
    ctx.metadata["denied_tool_calls"] = []
    ctx.metadata["consecutive_tool_errors"] = {"kit.recovered": 2}
    ctx.metadata["tool_results"] = [
        ToolResult(
            call_id="c1",
            status="ok",
            content=[ContentBlockOut(type="text", text="ok")],
            duration_ms=0,
            cached=False,
        )
    ]
    next_ = await handle_tool_results(ctx, {"max_consecutive_tool_errors": 3})
    assert next_ is Phase.CONTEXT_BUILD
    # Counter for the recovered tool was wiped.
    assert "kit.recovered" not in ctx.metadata["consecutive_tool_errors"]


@pytest.mark.asyncio
async def test_tool_phase_unknown_tool_routes_to_executing():
    """A hallucinated/unregistered tool name must still route through
    TOOL_EXECUTING so an 'unknown tool' result gets built. Routing straight
    to TOOL_RESULTS skips result construction, leaving the model with silence
    and no chance to self-correct."""
    reg = ToolRegistry()  # empty registry — no tools registered
    ctx = TurnContext.empty()
    ctx.metadata["pending_tool_calls"] = [{"id": "c1", "name": "get_kb_fact", "arguments": {}}]
    next_ = await handle_tool_phase(ctx, _make_deps(reg))
    assert next_ is Phase.TOOL_EXECUTING
    assert ctx.metadata["unknown_tool_calls"] == [
        {"id": "c1", "name": "get_kb_fact", "arguments": {}}
    ]
    # Unknown calls are not approval-denied — keep the categories distinct.
    assert ctx.metadata["denied_tool_calls"] == []


@pytest.mark.asyncio
async def test_tool_executing_builds_error_result_for_unknown_tool():
    """handle_tool_executing must synthesise a status='error' ToolResult for
    each unknown tool call, naming the bad tool so the model can self-correct."""
    reg = ToolRegistry()
    ctx = TurnContext.empty()
    ctx.metadata["approved_tool_calls"] = []
    ctx.metadata["denied_tool_calls"] = []
    ctx.metadata["unknown_tool_calls"] = [{"id": "c1", "name": "get_kb_fact", "arguments": {}}]
    next_ = await handle_tool_executing(ctx, _make_deps(reg))
    assert next_ is Phase.TOOL_RESULTS
    results = ctx.metadata["tool_results"]
    assert len(results) == 1
    assert results[0].call_id == "c1"
    assert results[0].status == "error"
    assert "unknown tool" in results[0].content[0].text
    assert "get_kb_fact" in results[0].content[0].text
    assert results[0].error is not None
    assert results[0].error.code == "unknown_tool"


@pytest.mark.asyncio
async def test_tool_results_counts_unknown_tool_errors_for_loop_abort():
    """F20: a model that keeps hallucinating the same unknown tool name must
    trip the consecutive-error abort instead of looping forever."""
    from agentkit.tools.spec import ToolError

    ctx = TurnContext.empty()
    ctx.metadata["approved_tool_calls"] = []
    ctx.metadata["denied_tool_calls"] = []
    ctx.metadata["unknown_tool_calls"] = [{"id": "c1", "name": "get_kb_fact", "arguments": {}}]
    ctx.metadata["tool_results"] = [
        ToolResult(
            call_id="c1",
            status="error",
            content=[],
            error=ToolError(code="unknown_tool", message="unknown tool: get_kb_fact"),
            duration_ms=0,
            cached=False,
        )
    ]
    ctx.metadata["consecutive_tool_errors"] = {"get_kb_fact": 2}  # this is the 3rd
    next_ = await handle_tool_results(ctx, {"max_consecutive_tool_errors": 3})
    assert next_ is Phase.ERRORED
    assert ctx.metadata["tool_error_loop"]["tool"] == "get_kb_fact"


@pytest.mark.asyncio
async def test_tool_results_carries_provenance_onto_the_persisted_block():
    """ToolResult.provenance must survive the fold into ctx.history.

    guards.taint.mark_taint reads ``ToolResult.provenance`` before this
    handler ever runs, so the taint decision itself is unaffected either way
    — but the ``ToolResultBlock`` this handler appends to ``ctx.history`` (and
    therefore to whatever the session store persists) has its own
    ``provenance`` field, defaulting to SYSTEM, and nothing carried the real
    value into it. A transcript re-read later — after a checkpoint resume, by
    an audit tool, or by any future code that infers trust from the stored
    message rather than re-deriving it — would see every result as SYSTEM
    regardless of where it actually came from.
    """
    from agentkit._content import Provenance, ToolResultBlock

    ctx = TurnContext.empty()
    ctx.metadata["tool_results"] = [
        ToolResult(
            call_id="c1",
            status="ok",
            content=[ContentBlockOut(type="text", text="scraped page text")],
            duration_ms=5,
            cached=False,
            provenance=Provenance.UNTRUSTED,
        )
    ]
    await handle_tool_results(ctx, {})

    block = ctx.history[-1].content[0]
    assert isinstance(block, ToolResultBlock)
    assert block.provenance is Provenance.UNTRUSTED


@pytest.mark.asyncio
async def test_tool_call_result_events_name_the_tool_that_produced_them():
    """A result event that carries only a call id makes every consumer keep a
    map back to ToolCallStarted — across reconnects and replays, where that
    earlier event may never arrive. The handler already builds this exact
    ``call_id -> name`` map for its consecutive-error counter.
    """
    ctx = TurnContext.empty()
    ctx.event_queue = asyncio.Queue()
    ctx.metadata["approved_tool_calls"] = [{"id": "c1", "name": "music_retag", "arguments": {}}]
    ctx.metadata["denied_tool_calls"] = [{"id": "c2", "name": "torrent_delete", "arguments": {}}]
    ctx.metadata["tool_results"] = [
        ToolResult(
            call_id="c1",
            status="ok",
            content=[ContentBlockOut(type="text", text="done")],
            error=None,
            duration_ms=1,
            cached=False,
        ),
        ToolResult(
            call_id="c2",
            status="denied",
            content=[ContentBlockOut(type="text", text="no")],
            error=None,
            duration_ms=1,
            cached=False,
        ),
        # Control: a result whose call was in none of the buckets. Proves the
        # names above were looked up rather than a constant, and that a miss
        # is visible as UNNAMED_TOOL instead of borrowing a neighbour's name.
        ToolResult(
            call_id="c-stray",
            status="ok",
            content=[ContentBlockOut(type="text", text="?")],
            error=None,
            duration_ms=1,
            cached=False,
        ),
    ]

    await handle_tool_results(ctx, {})

    events: list[ToolCallResult] = []
    while not ctx.event_queue.empty():
        ev = ctx.event_queue.get_nowait()
        if isinstance(ev, ToolCallResult):
            events.append(ev)

    assert {e.call_id: e.tool_name for e in events} == {
        "c1": "music_retag",
        "c2": "torrent_delete",
        "c-stray": UNNAMED_TOOL,
    }
