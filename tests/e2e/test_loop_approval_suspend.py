import asyncio

import pytest

from agentkit.events import ApprovalNeeded, TurnEnded
from agentkit.guards.approval import RiskBasedApprovalGate
from agentkit.guards.finalize import StructuralFinalizeValidator
from agentkit.loop.context import TurnContext
from agentkit.loop.message_builder import MessageBuilder
from agentkit.loop.orchestrator import Loop
from agentkit.loop.tool_dispatcher import DispatchPolicy, ToolDispatcher
from agentkit.providers.fakes import FakeProvider
from agentkit.tools.registry import ToolRegistry
from agentkit.tools.spec import (
    ApprovalPolicy,
    ContentBlockOut,
    RiskLevel,
    SideEffects,
    ToolResult,
    ToolSpec,
)
from tests.e2e.test_loop_text_only import _all_handlers, _user

pytestmark = pytest.mark.e2e


@pytest.mark.asyncio
async def test_high_write_call_suspends_for_approval():
    provider = FakeProvider().script(
        FakeProvider.tool_call("ampaera.devices.control", {"id": "heat-pump", "off": True}),
    )
    queue: asyncio.Queue = asyncio.Queue()
    registry = ToolRegistry()

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
        return ToolResult(
            call_id=ctx.call_id,
            status="ok",
            content=[ContentBlockOut(type="text", text="off")],
            error=None,
            duration_ms=1,
            cached=False,
        )

    registry.register_builtin(spec, handler)

    ctx = TurnContext.empty()
    ctx.event_queue = queue
    ctx.add_message(_user("turn off the heat pump"))

    deps = {
        "provider": provider,
        "message_builder": MessageBuilder(model="m", max_tokens=128),
        "registry": registry,
        "system_blocks": [],
        "intent_gate": None,
        "approval_gate": RiskBasedApprovalGate(),
        "dispatcher": ToolDispatcher(registry=registry, policy=DispatchPolicy()),
        "finalize_validator": StructuralFinalizeValidator(),
        "approval_timeout_seconds": 60,
        "max_finalize_retries": 2,
        "max_iterations": 10,
    }
    loop = Loop(ctx=ctx, handlers=_all_handlers(), deps=deps)

    events = [ev async for ev in loop.run()]
    assert isinstance(events[-1], TurnEnded)

    streamed = []
    while not queue.empty():
        streamed.append(queue.get_nowait())
    needed = [e for e in streamed if isinstance(e, ApprovalNeeded)]
    assert needed and needed[0].tool_name == "ampaera.devices.control"
