"""approval_wait: what the ApprovalNeeded card is told about the call."""

import asyncio

import pytest

from agentkit.events import ApprovalNeeded
from agentkit.events.approval import UNKNOWN_CLASSIFICATION
from agentkit.guards.taint import TaintSource
from agentkit.loop.context import TurnContext
from agentkit.loop.handlers.approval_wait import handle_approval_wait
from agentkit.loop.phase import Phase
from agentkit.tools.registry import ToolRegistry
from agentkit.tools.spec import (
    ApprovalPolicy,
    ContentBlockOut,
    RiskLevel,
    SideEffects,
    ToolResult,
    ToolSpec,
)


def _spec(name: str, side_effects: SideEffects) -> ToolSpec:
    return ToolSpec(
        name=name,
        description="d",
        parameters={"type": "object"},
        returns=None,
        risk=RiskLevel.HIGH_WRITE,
        idempotent=False,
        side_effects=side_effects,
        requires_approval=ApprovalPolicy.BY_RISK,
        cache_ttl_seconds=None,
        timeout_seconds=10.0,
    )


async def _noop(arguments: dict, ctx: TurnContext) -> ToolResult:
    return ToolResult(
        call_id=ctx.call_id,
        status="ok",
        content=[ContentBlockOut(type="text", text="ok")],
    )


def _ctx(*calls: dict) -> TurnContext:
    ctx = TurnContext.empty()
    ctx.event_queue = asyncio.Queue()
    ctx.metadata["pending_user_approvals"] = list(calls)
    return ctx


def _drain(ctx: TurnContext) -> list[ApprovalNeeded]:
    assert ctx.event_queue is not None
    out: list[ApprovalNeeded] = []
    while not ctx.event_queue.empty():
        out.append(ctx.event_queue.get_nowait())
    return out


@pytest.mark.asyncio
async def test_side_effects_comes_from_the_spec():
    registry = ToolRegistry()
    registry.register_builtin(_spec("vault.delete", SideEffects.EXTERNAL_IRREVERSIBLE), _noop)
    ctx = _ctx({"id": "c1", "name": "vault.delete", "arguments": {"email": "a@b.test"}})

    phase = await handle_approval_wait(ctx, {"registry": registry})

    assert phase is Phase.TURN_ENDED
    (ev,) = _drain(ctx)
    assert ev.risk == "high_write"
    # The value, not the enum member: this is what goes on the wire.
    assert ev.side_effects == "external_irreversible"
    assert isinstance(ev.model_dump(mode="json")["side_effects"], str)


@pytest.mark.asyncio
async def test_unregistered_tool_is_unknown_on_both_axes():
    """An unclassified tool must not be reported as harmless."""
    ctx = _ctx({"id": "c1", "name": "ghost.tool", "arguments": {}})

    await handle_approval_wait(ctx, {"registry": ToolRegistry()})

    (ev,) = _drain(ctx)
    assert ev.risk == UNKNOWN_CLASSIFICATION
    assert ev.side_effects == UNKNOWN_CLASSIFICATION


@pytest.mark.asyncio
async def test_taint_sources_are_carried_onto_every_card():
    registry = ToolRegistry()
    registry.register_builtin(_spec("vault.delete", SideEffects.EXTERNAL_IRREVERSIBLE), _noop)
    registry.register_builtin(_spec("torrent.add", SideEffects.EXTERNAL_REVERSIBLE), _noop)
    ctx = _ctx(
        {"id": "c1", "name": "vault.delete", "arguments": {}},
        {"id": "c2", "name": "torrent.add", "arguments": {}},
    )
    ctx.tainted = True
    ctx.taint_sources = [TaintSource(call_id="c0", tool_name="web.fetch", kind="untrusted")]

    await handle_approval_wait(ctx, {"registry": registry})

    events = _drain(ctx)
    assert len(events) == 2
    # Taint is a property of the turn, so both cards carry it.
    for ev in events:
        assert [s.tool_name for s in ev.taint] == ["web.fetch"]


@pytest.mark.asyncio
async def test_clean_turn_reports_no_taint():
    registry = ToolRegistry()
    registry.register_builtin(_spec("torrent.add", SideEffects.EXTERNAL_REVERSIBLE), _noop)
    ctx = _ctx({"id": "c1", "name": "torrent.add", "arguments": {}})

    await handle_approval_wait(ctx, {"registry": registry})

    (ev,) = _drain(ctx)
    assert ev.taint == []
