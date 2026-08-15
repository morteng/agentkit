"""ToolPlane resolution state is per-context, and the plane can gate execution.

Two regressions are pinned here:

* ``resolve`` used to write ``_last_rationale`` / ``_last_discoverable`` onto
  the shared plane instance, so two sessions resolving concurrently read each
  other's tiers;
* tier resolution shaped only what was *advertised*, so naming a hidden tool
  still ran it.
"""

import asyncio
from typing import Any, cast

from agentkit.loop.context import TurnContext
from agentkit.toolplane.plane import Resolution, ToolPlane, ToolPlaneAuthorizer
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

ROLE_RANKS = {"viewer": 0, "editor": 1, "admin": 2, "superuser": 3}


def _as_ctx(turn_ctx: object) -> ToolContext:
    return cast("ToolContext", turn_ctx)


def _spec(name: str, *, risk: RiskLevel = RiskLevel.READ) -> ToolSpec:
    return ToolSpec(
        name=name,
        description=f"desc for {name}",
        parameters={},
        returns=None,
        risk=risk,
        idempotent=True,
        side_effects=SideEffects.NONE,
        requires_approval=ApprovalPolicy.NEVER,
        cache_ttl_seconds=None,
        timeout_seconds=30.0,
    )


def _plane(vis_map: dict[str, ToolVisibility], context_of: Any = _as_ctx) -> ToolPlane:
    return ToolPlane(
        visibility_of=lambda spec: vis_map.get(spec.name),
        context_of=context_of,
        role_ranks=ROLE_RANKS,
    )


# ---- Statelessness ----------------------------------------------------------


def test_resolve_detailed_returns_everything_and_stores_nothing():
    specs = [_spec("srv.hot"), _spec("srv.buried")]
    plane = _plane({"srv.buried": ToolVisibility(baseline="discoverable")})

    resolution = plane.resolve_detailed(ToolContext(role="editor", role_rank=1), specs)

    assert isinstance(resolution, Resolution)
    assert [s.name for s in resolution.visible] == ["srv.hot"]
    assert [s.name for s in resolution.discoverable] == ["srv.buried"]
    assert resolution.rationale["srv.buried"].tier == "discoverable"
    # Pure: nothing was published to the plane.
    assert plane.rationale == {}
    assert plane.last_discoverable == []


def test_resolve_publishes_into_the_calling_context_only():
    specs = [_spec("srv.hot"), _spec("srv.buried")]
    plane = _plane({"srv.buried": ToolVisibility(baseline="discoverable")})

    visible = plane.resolve(ToolContext(role="editor", role_rank=1), specs)

    assert [s.name for s in visible] == ["srv.hot"]
    assert [s.name for s in plane.last_discoverable] == ["srv.buried"]
    assert plane.rationale["srv.hot"].tier == "hot"


async def test_concurrent_sessions_do_not_see_each_others_resolution():
    """The audit finding: per-call state on a shared instance cross-contaminates.

    Both tasks resolve before either reads, so an implementation that keeps the
    last resolution on the instance hands one session the other's tiers.
    """
    specs = [_spec("srv.alpha"), _spec("srv.beta")]
    plane = _plane(
        {
            "srv.alpha": ToolVisibility(baseline="discoverable"),
            "srv.beta": ToolVisibility(baseline="discoverable"),
        }
    )
    both_resolved = asyncio.Barrier(2)

    async def session(promoted: str) -> tuple[list[str], list[str]]:
        ctx = ToolContext(role="editor", role_rank=1, tier_overrides={promoted: "hot"})
        visible = plane.resolve(ctx, specs)
        await both_resolved.wait()
        # Read AFTER the other session has resolved.
        return [s.name for s in visible], [s.name for s in plane.last_discoverable]

    (a_visible, a_discoverable), (b_visible, b_discoverable) = await asyncio.wait_for(
        asyncio.gather(session("srv.alpha"), session("srv.beta")), timeout=5
    )

    assert a_visible == ["srv.alpha"]
    assert a_discoverable == ["srv.beta"]
    assert b_visible == ["srv.beta"]
    assert b_discoverable == ["srv.alpha"]


async def test_a_context_that_never_resolved_sees_an_empty_resolution():
    specs = [_spec("srv.buried")]
    plane = _plane({"srv.buried": ToolVisibility(baseline="discoverable")})

    async def resolver() -> None:
        plane.resolve(ToolContext(role="editor", role_rank=1), specs)
        assert plane.last_discoverable != []

    await asyncio.wait_for(asyncio.ensure_future(resolver()), timeout=5)

    # Sibling context: no resolution of its own, so nothing leaks in.
    async def observer() -> list[str]:
        return [s.name for s in plane.last_discoverable]

    assert await asyncio.wait_for(asyncio.ensure_future(observer()), timeout=5) == []


def test_two_planes_in_one_context_keep_separate_resolutions():
    specs = [_spec("srv.alpha"), _spec("srv.beta")]
    first = _plane({"srv.alpha": ToolVisibility(baseline="discoverable")})
    second = _plane({"srv.beta": ToolVisibility(baseline="discoverable")})
    ctx = ToolContext(role="editor", role_rank=1)

    first.resolve(ctx, specs)
    second.resolve(ctx, specs)

    assert [s.name for s in first.last_discoverable] == ["srv.alpha"]
    assert [s.name for s in second.last_discoverable] == ["srv.beta"]


# ---- Execution-time gate ----------------------------------------------------


def test_authorizer_denies_a_hidden_tool_and_allows_a_visible_one():
    hidden_spec = _spec("admin.purge", risk=RiskLevel.DESTRUCTIVE)
    plane = _plane({"admin.purge": ToolVisibility(min_role="admin")})
    authorizer = ToolPlaneAuthorizer(plane)

    viewer = ToolContext(role="viewer", role_rank=0)
    reason = authorizer.authorize(hidden_spec, viewer)
    assert reason is not None
    assert "admin.purge" in reason
    assert "min_role" in reason

    admin = ToolContext(role="admin", role_rank=2)
    assert authorizer.authorize(hidden_spec, admin) is None


def test_authorizer_allows_the_discoverable_tier_by_default():
    """Discoverable tools are reachable via search — merely not advertised."""
    spec = _spec("srv.buried")
    plane = _plane({"srv.buried": ToolVisibility(baseline="discoverable")})
    ctx = ToolContext(role="editor", role_rank=1)

    assert ToolPlaneAuthorizer(plane).authorize(spec, ctx) is None
    strict = ToolPlaneAuthorizer(plane, deny_tiers=frozenset({"hidden", "discoverable"}))
    assert strict.authorize(spec, ctx) is not None


async def test_registry_refuses_a_capability_gated_tool_at_execution_time():
    """The full path: the model names a tool the tenant has no capability for."""
    ran: list[str] = []

    async def handler(args: dict[str, Any], ctx: TurnContext) -> ToolResult:
        ran.append("purged")
        return ToolResult(
            call_id=ctx.call_id,
            status="ok",
            content=[ContentBlockOut(type="text", text="purged")],
        )

    spec = _spec("REDACTED.prune_facts", risk=RiskLevel.DESTRUCTIVE)
    plane = _plane(
        {"REDACTED.prune_facts": ToolVisibility(capability="kb_admin")},
        context_of=lambda turn_ctx: cast("TurnContext", turn_ctx).metadata["tool_context"],
    )
    reg = ToolRegistry(authorizer=ToolPlaneAuthorizer(plane))
    reg.register_builtin(spec, handler)

    without = TurnContext.empty()
    without.metadata["tool_context"] = ToolContext(
        role="editor", role_rank=1, capabilities=frozenset()
    )
    # It is not even advertised...
    assert plane.resolve(without, [spec]) == []
    # ...and naming it anyway does not run it.
    denied = await reg.invoke(ToolCall(id="c1", name=spec.name, arguments={}), without)
    assert denied.status == "denied"
    assert denied.error is not None
    assert denied.error.code == "not_authorized"
    assert ran == []

    granted = TurnContext.empty()
    granted.metadata["tool_context"] = ToolContext(
        role="editor", role_rank=1, capabilities=frozenset({"kb_admin"})
    )
    allowed = await reg.invoke(ToolCall(id="c2", name=spec.name, arguments={}), granted)
    assert allowed.status == "ok"
    assert ran == ["purged"]
