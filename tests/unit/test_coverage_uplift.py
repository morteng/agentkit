"""Targeted tests added during pre-release validation (Task 45) to lift
coverage of the modules the cov-fail-under threshold is meant to enforce
(loop/, guards/, tools/registry).

Each test is small and named after the missed branch it covers. Grouped here
rather than spread across many files because they were added together as a
coverage pass; future contributors should feel free to migrate any individual
test to the more specific module test file when touching that module.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from agentkit._content import TextBlock
from agentkit._ids import MessageId, SessionId, new_id
from agentkit._messages import Message, MessageRole, Usage
from agentkit.errors import ToolError as ToolErr
from agentkit.events import PhaseChanged, ToolCallResult, TurnEnded, TurnEndReason
from agentkit.guards.approval import RiskBasedApprovalGate
from agentkit.guards.finalize import StructuralFinalizeValidator
from agentkit.guards.success_claim import RegexSuccessClaimGuard
from agentkit.loop.context import FixedClock, TurnContext
from agentkit.loop.handlers.errored import handle_errored
from agentkit.loop.handlers.finalize_check import handle_finalize_check
from agentkit.loop.handlers.streaming import handle_streaming
from agentkit.loop.handlers.tool_executing import handle_tool_executing
from agentkit.loop.handlers.tool_phase import handle_tool_phase
from agentkit.loop.handlers.tool_results import handle_tool_results
from agentkit.loop.handlers.turn_ended import handle_turn_ended
from agentkit.loop.message_builder import MessageBuilder
from agentkit.loop.orchestrator import Loop
from agentkit.loop.phase import Phase
from agentkit.loop.stream_mux import StreamMux
from agentkit.loop.tool_dispatcher import DispatchPolicy, ToolDispatcher
from agentkit.mcp_client.inprocess import InProcessMCPClient
from agentkit.providers.base import (
    ErrorEvent,
    MessageStart,
    ThinkingDelta,
    ToolCallDelta,
    UsageEvent,
)
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

# ---- helpers ---------------------------------------------------------------


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


async def _ok_handler(args: dict[str, Any]) -> ToolResult:
    return ToolResult(
        call_id="",
        status="ok",
        content=[ContentBlockOut(type="text", text="hi")],
        error=None,
        duration_ms=0,
        cached=False,
    )


def _user(text: str) -> Message:
    return Message(
        id=new_id(MessageId),
        session_id=new_id(SessionId),
        role=MessageRole.USER,
        content=[TextBlock(text=text)],
        created_at=datetime.now(UTC),
    )


# ---- ToolRegistry: MCP integration paths -----------------------------------


@pytest.mark.asyncio
async def test_registry_initialize_and_invoke_mcp_tool() -> None:
    """register_mcp_server -> initialize_mcp_servers -> list_specs -> invoke."""
    reg = ToolRegistry()
    server = InProcessMCPClient(name="srv")
    server.register_tool(_spec("ping"), _ok_handler)
    reg.register_mcp_server("srv", server)

    await reg.initialize_mcp_servers()

    specs = reg.list_specs()
    qualified = [s.name for s in specs]
    assert "srv.ping" in qualified

    result = await reg.invoke(
        ToolCall(id="c1", name="srv.ping", arguments={}),
        ctx=TurnContext.empty(),
    )
    assert result.status == "ok"

    await reg.shutdown()


@pytest.mark.asyncio
async def test_registry_rejects_duplicate_mcp_server() -> None:
    reg = ToolRegistry()
    reg.register_mcp_server("srv", InProcessMCPClient(name="srv"))
    with pytest.raises(ToolErr):
        reg.register_mcp_server("srv", InProcessMCPClient(name="srv"))


@pytest.mark.asyncio
async def test_registry_rejects_namespace_collision_with_builtin() -> None:
    reg = ToolRegistry()

    async def bh(args: dict[str, Any], ctx: TurnContext) -> ToolResult:
        return ToolResult(
            call_id=ctx.call_id,
            status="ok",
            content=[],
            error=None,
            duration_ms=0,
            cached=False,
        )

    reg.register_builtin(_spec("srv.ping"), bh)
    server = InProcessMCPClient(name="srv")
    server.register_tool(_spec("ping"), _ok_handler)
    reg.register_mcp_server("srv", server)

    with pytest.raises(ToolErr):
        await reg.initialize_mcp_servers()


@pytest.mark.asyncio
async def test_registry_rejects_duplicate_mcp_tool_on_reinit() -> None:
    """Calling initialize_mcp_servers twice on the same registry collides."""
    reg = ToolRegistry()
    server = InProcessMCPClient(name="a")
    server.register_tool(_spec("ping"), _ok_handler)
    reg.register_mcp_server("a", server)

    await reg.initialize_mcp_servers()

    with pytest.raises(ToolErr):
        await reg.initialize_mcp_servers()


@pytest.mark.asyncio
async def test_registry_invoke_unknown_server_for_qualified_tool() -> None:
    """Qualified tool name registered, but server entry was deleted under us."""
    reg = ToolRegistry()
    server = InProcessMCPClient(name="srv")
    server.register_tool(_spec("ping"), _ok_handler)
    reg.register_mcp_server("srv", server)
    await reg.initialize_mcp_servers()

    # Forcibly drop the server while keeping the spec — invocation should error.
    reg._mcp_servers.pop("srv")  # type: ignore[attr-defined]

    with pytest.raises(ToolErr):
        await reg.invoke(ToolCall(id="c1", name="srv.ping", arguments={}), ctx=TurnContext.empty())


# ---- Loop orchestrator: error paths ----------------------------------------


@pytest.mark.asyncio
async def test_loop_routes_to_errored_when_handler_missing() -> None:
    """Starting phase has no handler -> ERRORED + TurnEnded(ERROR)."""
    ctx = TurnContext.empty()
    loop = Loop(ctx=ctx, handlers={}, end_reason=TurnEndReason.COMPLETED)
    events = [ev async for ev in loop.run()]
    assert isinstance(events[-1], TurnEnded)
    assert events[-1].reason is TurnEndReason.ERROR
    pc = [e for e in events if isinstance(e, PhaseChanged)]
    assert pc[-1].to is Phase.ERRORED


@pytest.mark.asyncio
async def test_loop_handles_handler_exception() -> None:
    async def bad(ctx: TurnContext, deps: dict[str, Any]) -> Phase:
        raise RuntimeError("boom")

    ctx = TurnContext.empty()
    loop = Loop(
        ctx=ctx,
        handlers={Phase.INTENT_GATE: bad},
        end_reason=TurnEndReason.COMPLETED,
    )
    events = [ev async for ev in loop.run()]
    assert isinstance(events[-1], TurnEnded)
    assert events[-1].reason is TurnEndReason.ERROR
    pc = [e for e in events if isinstance(e, PhaseChanged)]
    assert pc[-1].to is Phase.ERRORED


@pytest.mark.asyncio
async def test_loop_rejects_invalid_transition() -> None:
    """Handler returns a phase that's not in the transition table."""

    async def bad_transition(ctx: TurnContext, deps: dict[str, Any]) -> Phase:
        # INTENT_GATE -> TOOL_EXECUTING is not a legal transition
        return Phase.TOOL_EXECUTING

    ctx = TurnContext.empty()
    loop = Loop(
        ctx=ctx,
        handlers={Phase.INTENT_GATE: bad_transition},
        end_reason=TurnEndReason.COMPLETED,
    )
    events = [ev async for ev in loop.run()]
    assert isinstance(events[-1], TurnEnded)
    assert events[-1].reason is TurnEndReason.ERROR


# ---- Terminal handlers (trivial) -------------------------------------------


@pytest.mark.asyncio
async def test_handle_errored_returns_errored_phase() -> None:
    ctx = TurnContext.empty()
    assert await handle_errored(ctx, {}) is Phase.ERRORED


@pytest.mark.asyncio
async def test_handle_turn_ended_returns_turn_ended_phase() -> None:
    ctx = TurnContext.empty()
    assert await handle_turn_ended(ctx, {}) is Phase.TURN_ENDED


def test_turn_end_module_re_exports_turn_end_reason() -> None:
    from agentkit.loop.turn_end import TurnEndReason as Re

    assert Re is TurnEndReason


# ---- Loop handler: tool_phase auto_deny path -------------------------------


@pytest.mark.asyncio
async def test_tool_phase_auto_deny_when_spec_missing() -> None:
    """Pending call references a tool name not in the registry -> auto_deny."""
    reg = ToolRegistry()  # empty
    deps = {
        "registry": reg,
        "approval_gate": RiskBasedApprovalGate(),
        "dispatcher": ToolDispatcher(registry=reg, policy=DispatchPolicy()),
    }

    ctx = TurnContext.empty()
    ctx.metadata["pending_tool_calls"] = [
        {"id": "c1", "name": "kit.does_not_exist", "arguments": {}}
    ]
    next_ = await handle_tool_phase(ctx, deps)
    assert next_ is Phase.TOOL_RESULTS
    assert ctx.metadata["denied_tool_calls"] and not ctx.metadata["approved_tool_calls"]


# ---- Loop handler: tool_executing denied path ------------------------------


@pytest.mark.asyncio
async def test_tool_executing_synthesizes_denied_results() -> None:
    reg = ToolRegistry()
    deps = {
        "registry": reg,
        "approval_gate": RiskBasedApprovalGate(),
        "dispatcher": ToolDispatcher(registry=reg, policy=DispatchPolicy()),
    }

    ctx = TurnContext.empty()
    ctx.metadata["approved_tool_calls"] = []
    ctx.metadata["denied_tool_calls"] = [{"id": "c1", "name": "kit.x", "arguments": {}}]
    next_ = await handle_tool_executing(ctx, deps)
    assert next_ is Phase.TOOL_RESULTS
    results = ctx.metadata["tool_results"]
    assert len(results) == 1 and results[0].status == "denied"


# ---- Loop handler: tool_results emits events + max-iter guard --------------


@pytest.mark.asyncio
async def test_tool_results_emits_event_and_appends_history() -> None:
    import asyncio

    ctx = TurnContext.empty()
    ctx.event_queue = asyncio.Queue()
    ctx.metadata["tool_results"] = [
        ToolResult(
            call_id="c1",
            status="ok",
            content=[ContentBlockOut(type="text", text="result body")],
            error=None,
            duration_ms=5,
            cached=False,
        )
    ]
    ctx.finalize_called = True
    next_ = await handle_tool_results(ctx, {})
    assert next_ is Phase.FINALIZE_CHECK
    # Event should have been pushed to the queue.
    ev = await ctx.event_queue.get()
    assert isinstance(ev, ToolCallResult)
    assert ev.call_id == "c1"


@pytest.mark.asyncio
async def test_tool_results_routes_to_finalize_check_when_max_iterations_hit() -> None:
    ctx = TurnContext.empty()
    ctx.metadata["tool_results"] = []
    ctx.metadata["iterations"] = 9  # next iteration will be 10
    next_ = await handle_tool_results(ctx, {"max_iterations": 10})
    assert next_ is Phase.FINALIZE_CHECK
    assert ctx.metadata["max_iterations_hit"] is True


# ---- Loop handler: finalize_check exhausted-retries path -------------------


@pytest.mark.asyncio
async def test_finalize_check_routes_to_memory_when_retries_exhausted() -> None:
    ctx = TurnContext.empty()
    ctx.add_message(_user("turn off the heat pump"))
    ctx.finalize_called = True
    ctx.metadata["finalize_retries"] = 2
    deps = {"finalize_validator": StructuralFinalizeValidator(), "max_finalize_retries": 2}
    next_ = await handle_finalize_check(ctx, deps)
    assert next_ is Phase.MEMORY_EXTRACT
    assert ctx.metadata["finalize_exhausted"] is True


# ---- Guards: finalize edge cases -------------------------------------------


@pytest.mark.asyncio
async def test_finalize_validator_accepts_valid_answer_envelope() -> None:
    """Structural validator accepts a well-formed answer envelope with no writes."""
    v = StructuralFinalizeValidator()
    ctx = TurnContext.empty()  # no history
    valid_args = {"status": "done", "intent_kind": "answer", "actions_performed": []}
    verdict = await v.validate(
        ToolCall(id="finalize", name="kit.finalize", arguments=valid_args), ctx
    )
    assert verdict.accept is True


# ---- Guards: success_claim no-flag path ------------------------------------


@pytest.mark.asyncio
async def test_success_claim_no_flag_when_no_pattern_match() -> None:
    g = RegexSuccessClaimGuard()
    ctx = TurnContext.empty()
    verdict = await g.check("everything is fine", ctx)
    assert verdict.flag is False


@pytest.mark.asyncio
async def test_success_claim_flags_when_no_non_kit_tool_called() -> None:
    g = RegexSuccessClaimGuard()
    ctx = TurnContext.empty()
    verdict = await g.check("all set", ctx)
    assert verdict.flag is True


@pytest.mark.asyncio
async def test_success_claim_no_flag_when_non_kit_tool_was_called() -> None:
    """If the assistant actually invoked a real (non-kit) tool, don't flag."""
    from agentkit._content import ToolUseBlock

    g = RegexSuccessClaimGuard()
    ctx = TurnContext.empty()
    # Simulate prior assistant message with a non-kit tool call.
    ctx.add_message(
        Message(
            id=new_id(MessageId),
            session_id=ctx.session_id,
            role=MessageRole.ASSISTANT,
            content=[ToolUseBlock(id="t1", name="device.toggle", arguments={})],
            created_at=datetime.now(UTC),
        )
    )
    verdict = await g.check("all set", ctx)
    assert verdict.flag is False


# ---- StreamMux: thinking, tool_call_delta, error branches ------------------


@pytest.mark.asyncio
async def test_stream_mux_thinking_and_error_branches() -> None:
    ctx = TurnContext.empty(clock=FixedClock(datetime.now(UTC)))
    mux = StreamMux(ctx)

    # Touch the sequence property at least once for coverage of the getter body.
    assert mux.sequence == 0

    async def src():
        yield MessageStart()
        yield ThinkingDelta(delta="hmm")
        # tool_call_delta is a no-op pass branch (not surfaced)
        yield ToolCallDelta(call_id="c1", arguments_delta='{"a"')
        yield UsageEvent(usage=Usage(input_tokens=1, output_tokens=1))
        yield ErrorEvent(code="provider_fault", message="oops", recoverable=True)

    out = [e async for e in mux.translate(src())]
    type_names = [type(e).__name__ for e in out]
    assert "ThinkingDelta" in type_names
    assert "Errored" in type_names


# ---- Streaming handler: success_claim flag short-circuits ------------------


@pytest.mark.asyncio
async def test_streaming_handler_short_circuits_on_success_claim_flag() -> None:
    """When SuccessClaimGuard flags during streaming, the handler returns
    Phase.CONTEXT_BUILD with a correction stashed in metadata.
    """
    import asyncio

    from agentkit.providers.base import (
        MessageComplete,
        MessageStart,
        TextDelta,
    )

    class _StubProvider:
        async def stream(self, request):  # type: ignore[no-untyped-def]
            yield MessageStart()
            yield TextDelta(delta="all set")
            yield MessageComplete(finish_reason="end_turn")

    reg = ToolRegistry()
    builder = MessageBuilder(model="x", max_tokens=4096)
    ctx = TurnContext.empty()
    ctx.event_queue = asyncio.Queue()
    deps: dict[str, Any] = {
        "provider": _StubProvider(),
        "message_builder": builder,
        "registry": reg,
        "success_claim": RegexSuccessClaimGuard(),
        "system_blocks": [],
    }
    next_ = await handle_streaming(ctx, deps)
    assert next_ is Phase.CONTEXT_BUILD
    assert "claim_correction" in ctx.metadata
