"""F18: AgentSession wires SubagentDispatcher into deps and tool_executing
injects ctx.spawn_subagent before invoking handlers."""

import asyncio

import pytest

from agentkit import AgentConfig, AgentSession
from agentkit._ids import OwnerId
from agentkit.guards.approval import RiskBasedApprovalGate
from agentkit.loop.context import TurnContext
from agentkit.loop.handlers.tool_executing import handle_tool_executing
from agentkit.loop.tool_dispatcher import DispatchPolicy, ToolDispatcher
from agentkit.providers.fakes import FakeProvider
from agentkit.store.fakes import FakeCheckpointStore, FakeMemoryStore, FakeSessionStore
from agentkit.subagents.dispatcher import SubagentDispatcher
from agentkit.tools.registry import ToolRegistry
from agentkit.tools.spec import (
    ApprovalPolicy,
    ContentBlockOut,
    RiskLevel,
    SideEffects,
    ToolResult,
    ToolSpec,
)


def _make_session() -> AgentSession:
    config = AgentConfig()
    config.guards.approval = RiskBasedApprovalGate()
    config.stores.session = FakeSessionStore()
    config.stores.memory = FakeMemoryStore()
    config.stores.checkpoint = FakeCheckpointStore()
    registry = ToolRegistry()
    registry.register_default_builtins()
    return AgentSession(
        owner=OwnerId("u"),
        config=config,
        provider=FakeProvider(),
        registry=registry,
        model="fake",
    )


def test_build_deps_includes_subagent_dispatcher():
    session = _make_session()
    deps = session._build_deps()
    assert "subagent_dispatcher" in deps
    assert isinstance(deps["subagent_dispatcher"], SubagentDispatcher)


def test_build_deps_subagent_max_depth_matches_config():
    session = _make_session()
    session.config.loop.max_subagent_depth = 7
    deps = session._build_deps()
    sub: SubagentDispatcher = deps["subagent_dispatcher"]
    assert sub._max_depth == 7


@pytest.mark.asyncio
async def test_tool_executing_injects_spawn_subagent_when_dispatcher_present():
    """If deps carry a subagent_dispatcher, ctx.spawn_subagent is set up before tools run."""
    registry = ToolRegistry()

    spy: dict[str, object] = {}

    spec = ToolSpec(
        name="probe.has_spawn",
        description="record whether ctx.spawn_subagent was injected",
        parameters={"type": "object"},
        returns=None,
        risk=RiskLevel.READ,
        idempotent=True,
        side_effects=SideEffects.NONE,
        requires_approval=ApprovalPolicy.NEVER,
        cache_ttl_seconds=None,
        timeout_seconds=10.0,
    )

    async def handler(_args, ctx):
        spy["spawn_subagent_present"] = ctx.spawn_subagent is not None
        return ToolResult(
            call_id=ctx.call_id,
            status="ok",
            content=[ContentBlockOut(type="text", text="ok")],
        )

    registry.register_builtin(spec, handler)

    fake_dispatcher = SubagentDispatcher(deps={}, max_depth=2)
    ctx = TurnContext.empty()
    ctx.event_queue = asyncio.Queue()
    ctx.metadata["approved_tool_calls"] = [{"id": "c1", "name": "probe.has_spawn", "arguments": {}}]
    ctx.metadata["denied_tool_calls"] = []

    deps = {
        "registry": registry,
        "dispatcher": ToolDispatcher(registry=registry, policy=DispatchPolicy()),
        "subagent_dispatcher": fake_dispatcher,
    }

    await handle_tool_executing(ctx, deps)

    assert spy["spawn_subagent_present"] is True
    assert ctx.spawn_subagent is not None  # left injected on ctx after dispatch
