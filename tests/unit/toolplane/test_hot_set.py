from typing import cast

from agentkit.toolplane import ToolPlane
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


def _superuser_ctx() -> ToolContext:
    return ToolContext(role="superuser", role_rank=3)


def test_hot_set_returns_only_baseline_hot_tools():
    vis = {
        "a": ToolVisibility(baseline="hot"),
        "b": ToolVisibility(baseline="discoverable"),
        "c": ToolVisibility(baseline="discoverable", pages=["/x"]),  # not promoted w/o page
    }
    plane = _plane(vis)
    specs = [_spec("a"), _spec("b"), _spec("c"), _spec("d")]  # d -> default hot
    assert plane.hot_set(specs, _superuser_ctx()) == {"a", "d"}


def test_hot_set_excludes_min_role_gated_below_role():
    vis = {"admin_only": ToolVisibility(baseline="hot", min_role="superuser")}
    plane = _plane(vis)
    specs = [_spec("admin_only")]
    editor = ToolContext(role="editor", role_rank=1)
    assert plane.hot_set(specs, editor) == set()
    assert plane.hot_set(specs, _superuser_ctx()) == {"admin_only"}
