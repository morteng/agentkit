"""AgentSession must actually connect the knobs it advertises.

Every failure pinned here is the same shape: a mechanism was built and tested
in isolation, and nothing wired it into the session — so the guard existed, the
config field existed, and the deployment was still unprotected. Unit tests on
the mechanisms cannot catch that. These tests assert the wiring itself.
"""

from typing import Any, cast

import pytest

from agentkit import AgentConfig, AgentSession
from agentkit._ids import OwnerId
from agentkit.guards.intent import DEFAULT_TURNS_PER_MINUTE
from agentkit.loop.context import TurnContext
from agentkit.providers.fakes import FakeProvider
from agentkit.store.fakes import FakeSessionStore
from agentkit.toolplane.plane import ToolPlane
from agentkit.toolplane.types import ToolContext, ToolVisibility
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


def _spec(name: str, *, risk: RiskLevel = RiskLevel.READ) -> ToolSpec:
    return ToolSpec(
        name=name,
        description=name,
        parameters={"type": "object"},
        returns=None,
        risk=risk,
        idempotent=True,
        side_effects=SideEffects.NONE,
        requires_approval=ApprovalPolicy.NEVER,
        cache_ttl_seconds=None,
        timeout_seconds=30.0,
    )


def _session(config: AgentConfig, registry: ToolRegistry) -> AgentSession:
    config.stores.session = FakeSessionStore()
    return AgentSession(
        owner=OwnerId("u:1"),
        config=config,
        provider=FakeProvider(),
        registry=registry,
        model="m",
    )


@pytest.mark.asyncio
async def test_a_configured_tool_plane_gates_execution_not_just_advertising():
    """Naming a hidden tool must not run it.

    ``ToolPlane`` filtering the advertised catalog is advisory — a model can
    name a tool it was never shown, from an earlier turn or from an injected
    instruction. Building ``ToolPlaneAuthorizer`` but leaving it unwired means
    the plane still only shapes advertising, which is the finding it was
    written to close.
    """
    ran: list[str] = []

    async def handler(args: dict[str, Any], ctx: TurnContext) -> ToolResult:
        ran.append("purge")
        return ToolResult(
            call_id=ctx.call_id,
            status="ok",
            content=[ContentBlockOut(type="text", text="purged")],
        )

    spec = _spec("admin.purge", risk=RiskLevel.DESTRUCTIVE)
    registry = ToolRegistry()
    registry.register_builtin(spec, handler)

    config = AgentConfig()
    config.tool_selector = ToolPlane(
        visibility_of=lambda s: ToolVisibility(min_role="admin") if s.name == spec.name else None,
        context_of=lambda turn_ctx: cast("TurnContext", turn_ctx).metadata["tool_context"],
        role_ranks={"viewer": 0, "admin": 2},
    )
    _session(config, registry)

    ctx = TurnContext.empty()
    ctx.metadata["tool_context"] = ToolContext(role="viewer", role_rank=0, capabilities=frozenset())

    denied = await registry.invoke(ToolCall(id="c1", name=spec.name, arguments={}), ctx)
    assert denied.status == "denied"
    assert denied.error is not None
    assert denied.error.code == "not_authorized"
    assert ran == []

    admin = TurnContext.empty()
    admin.metadata["tool_context"] = ToolContext(
        role="admin", role_rank=2, capabilities=frozenset()
    )
    assert (
        await registry.invoke(ToolCall(id="c2", name=spec.name, arguments={}), admin)
    ).status == "ok"
    assert ran == ["purge"]


@pytest.mark.asyncio
async def test_a_consumer_authorizer_is_preserved_and_not_nested_per_session():
    """The session composes with your gate; N sessions do not stack N chains.

    Reading ``registry.authorizer`` and wrapping whatever is there is the
    obvious implementation and the wrong one: on a registry shared by many
    sessions each new session would wrap the previous session's chain, so the
    chain grows without bound and every call walks one layer per session ever
    constructed. The invariant is that the session always rebuilds from the
    *consumer's* authorizer, which is what ``base`` identity checks.
    """
    calls: list[str] = []

    class CountingAuthorizer:
        def authorize(self, spec: ToolSpec, ctx: Any) -> str | None:
            calls.append(spec.name)
            return None

    consumer = CountingAuthorizer()
    registry = ToolRegistry(authorizer=consumer)

    async def ping(args: dict[str, Any], ctx: TurnContext) -> ToolResult:
        return ToolResult(call_id=ctx.call_id, status="ok", content=[])

    registry.register_builtin(_spec("srv.ping"), ping)

    for _ in range(5):
        _session(AgentConfig(), registry)

    installed = registry.authorizer
    assert installed is not consumer, "the session did not install its own chain"
    # Rebuilt from the consumer's gate every time, never from the last chain.
    assert getattr(installed, "base", None) is consumer

    result = await registry.invoke(
        ToolCall(id="c1", name="srv.ping", arguments={}), TurnContext.empty()
    )
    assert result.status == "ok"
    assert calls == ["srv.ping"], "the consumer gate still runs, exactly once"


def test_the_rate_limiter_reaches_the_loop():
    """``GuardConfig.rate_limit_turns_per_minute`` must arrive as a real gate.

    It defaulted to 60 while ``_build_deps`` still passed the raw
    ``guards.intent`` (usually ``None``), so the limit was configuration with
    nothing behind it.
    """
    session = _session(AgentConfig(), ToolRegistry())
    gate = session._build_deps()["intent_gate"]  # pyright: ignore[reportPrivateUsage]
    assert gate is not None

    # ...and the same instance every turn: the sliding window lives on the
    # limiter, so a gate rebuilt per turn would forget every turn it counted.
    assert session._build_deps()["intent_gate"] is gate  # pyright: ignore[reportPrivateUsage]

    disabled = AgentConfig()
    disabled.guards.rate_limit_turns_per_minute = None
    assert _session(disabled, ToolRegistry())._build_deps()["intent_gate"] is None  # pyright: ignore[reportPrivateUsage]

    assert DEFAULT_TURNS_PER_MINUTE == 60


def test_loop_knobs_reach_the_handlers():
    """Three deps that existed as config fields and arrived nowhere."""
    config = AgentConfig()
    config.loop.max_claim_corrections = 7
    config.loop.streaming_chunk_timeout_seconds = 3.5
    deps = _session(config, ToolRegistry())._build_deps()  # pyright: ignore[reportPrivateUsage]

    assert deps["max_claim_corrections"] == 7
    assert deps["streaming_chunk_timeout_seconds"] == 3.5
