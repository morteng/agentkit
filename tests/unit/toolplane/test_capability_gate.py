from typing import cast

from agentkit.toolplane import ToolPlane, tool_capability_satisfied
from agentkit.toolplane.types import ToolContext, ToolVisibility
from agentkit.tools.spec import ApprovalPolicy, RiskLevel, SideEffects, ToolSpec

ROLE_RANKS = {"viewer": 0, "editor": 1, "admin": 2, "superuser": 3}


def _as_ctx(turn_ctx: object) -> ToolContext:
    return cast("ToolContext", turn_ctx)


def _spec(name: str) -> ToolSpec:
    return ToolSpec(
        name=name,
        description=f"desc for {name}",
        parameters={},
        returns=None,
        risk=RiskLevel.READ,
        idempotent=True,
        side_effects=SideEffects.NONE,
        requires_approval=ApprovalPolicy.NEVER,
        cache_ttl_seconds=None,
        timeout_seconds=30.0,
    )


def _plane(vis_map):
    return ToolPlane(
        visibility_of=lambda spec: vis_map.get(spec.name),
        context_of=_as_ctx,
        role_ranks=ROLE_RANKS,
    )


def test_predicate_untagged_tool_always_satisfied():
    assert tool_capability_satisfied(ToolVisibility(), frozenset()) is True
    assert tool_capability_satisfied(ToolVisibility(capability="x"), frozenset({"x"})) is True
    assert tool_capability_satisfied(ToolVisibility(capability="x"), frozenset()) is False


def test_capability_missing_hides_tool():
    vis = {"pikkolo.create_menu": ToolVisibility(baseline="hot", capability="restaurant_menu")}
    plane = _plane(vis)
    specs = [_spec("pikkolo.create_menu")]
    ctx = ToolContext(role="editor", role_rank=1, capabilities=frozenset())
    out = plane.resolve(ctx, specs)
    assert out == []
    assert plane.rationale["pikkolo.create_menu"].tier == "hidden"
    assert "capability" in plane.rationale["pikkolo.create_menu"].reason


def test_capability_present_keeps_tool_visible():
    vis = {"pikkolo.create_menu": ToolVisibility(baseline="hot", capability="restaurant_menu")}
    plane = _plane(vis)
    specs = [_spec("pikkolo.create_menu")]
    ctx = ToolContext(role="editor", role_rank=1, capabilities=frozenset({"restaurant_menu"}))
    out = plane.resolve(ctx, specs)
    assert [s.name for s in out] == ["pikkolo.create_menu"]
    assert plane.rationale["pikkolo.create_menu"].tier == "hot"


def test_capability_gate_beats_page_promotion():
    vis = {
        "pikkolo.create_menu": ToolVisibility(
            baseline="discoverable", pages=["/dashboard/restaurant/*"], capability="restaurant_menu"
        )
    }
    plane = _plane(vis)
    specs = [_spec("pikkolo.create_menu")]
    ctx = ToolContext(
        role="editor",
        role_rank=1,
        page_path="/dashboard/restaurant/menus",
        capabilities=frozenset(),
    )
    out = plane.resolve(ctx, specs)
    assert out == []
    assert plane.rationale["pikkolo.create_menu"].tier == "hidden"
