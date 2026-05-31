from typing import cast

from agentkit.toolplane import ToolPlane
from agentkit.toolplane.types import ToolContext, ToolDecision, ToolVisibility
from agentkit.tools.spec import ApprovalPolicy, RiskLevel, SideEffects, ToolSpec

ROLE_RANKS = {"viewer": 0, "editor": 1, "admin": 2, "superuser": 3}


def _as_ctx(turn_ctx: object) -> ToolContext:
    """Test helper: callers pass a ToolContext directly; cast confirms the type."""
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


def _plane(vis_map, rules=None):
    return ToolPlane(
        visibility_of=lambda spec: vis_map.get(spec.name),
        context_of=_as_ctx,
        role_ranks=ROLE_RANKS,
        rules=rules,
    )


def test_unannotated_tool_is_hot():
    specs = [_spec("pikkolo.search")]
    plane = _plane({})
    out = plane.resolve(ToolContext(role="editor", role_rank=1), specs)
    assert [s.name for s in out] == ["pikkolo.search"]
    assert plane.rationale["pikkolo.search"].tier == "hot"


def test_discoverable_baseline_is_dropped_from_array():
    specs = [_spec("pikkolo.csg_subtract")]
    plane = _plane({"pikkolo.csg_subtract": ToolVisibility(baseline="discoverable")})
    out = plane.resolve(ToolContext(role="editor", role_rank=1), specs)
    assert out == []
    assert plane.rationale["pikkolo.csg_subtract"].tier == "discoverable"


def test_page_match_promotes_discoverable_to_active():
    specs = [_spec("pikkolo.csg_subtract")]
    plane = _plane(
        {
            "pikkolo.csg_subtract": ToolVisibility(
                baseline="discoverable", pages=["/dashboard/skins/*"]
            )
        }
    )
    ctx = ToolContext(role="editor", role_rank=1, page_path="/dashboard/skins/abc")
    out = plane.resolve(ctx, specs)
    assert [s.name for s in out] == ["pikkolo.csg_subtract"]
    assert plane.rationale["pikkolo.csg_subtract"].tier == "active"
    assert "page match" in plane.rationale["pikkolo.csg_subtract"].reason


def test_min_role_hard_gate_hides_tool():
    specs = [_spec("pikkolo.prune_facts")]
    plane = _plane({"pikkolo.prune_facts": ToolVisibility(baseline="hot", min_role="admin")})
    ctx = ToolContext(role="editor", role_rank=1)
    out = plane.resolve(ctx, specs)
    assert out == []
    assert plane.rationale["pikkolo.prune_facts"].tier == "hidden"


def test_intent_keyword_whole_word_match():
    specs = [_spec("pikkolo.csg_subtract")]
    plane = _plane(
        {
            "pikkolo.csg_subtract": ToolVisibility(
                baseline="discoverable", intent_keywords=["csg", "3d"]
            )
        }
    )
    hit = ToolContext(role="editor", role_rank=1, recent_user_message="make a 3d model please")
    miss = ToolContext(role="editor", role_rank=1, recent_user_message="threendimensional")
    assert [s.name for s in plane.resolve(hit, specs)] == ["pikkolo.csg_subtract"]
    assert plane.resolve(miss, specs) == []


def test_discovered_set_promotes_to_active():
    specs = [_spec("pikkolo.csg_subtract")]
    plane = _plane({"pikkolo.csg_subtract": ToolVisibility(baseline="discoverable")})
    ctx = ToolContext(role="editor", role_rank=1, discovered_tools=frozenset({"csg_subtract"}))
    assert [s.name for s in plane.resolve(ctx, specs)] == ["pikkolo.csg_subtract"]


def test_pluggable_rule_overrides_declarative():
    specs = [_spec("pikkolo.prune_facts")]
    rules = {"prune_facts": lambda ctx: ToolDecision("hidden", "rule: too few facts")}
    plane = _plane({"pikkolo.prune_facts": ToolVisibility(baseline="hot")}, rules=rules)
    out = plane.resolve(ToolContext(role="admin", role_rank=2), specs)
    assert out == []
    assert "rule:" in plane.rationale["pikkolo.prune_facts"].reason


def test_active_cap_enforced():
    specs = [_spec(f"pikkolo.t{i}") for i in range(40)]
    vis = {s.name: ToolVisibility(baseline="active") for s in specs}
    plane = _plane(vis)
    out = plane.resolve(ToolContext(role="editor", role_rank=1), specs)
    assert len(out) <= ToolPlane.ACTIVE_CAP


def test_hot_tier_not_truncated():
    # 40 hot tools must all survive — hot is uncapped during migration.
    specs = [_spec(f"pikkolo.h{i}") for i in range(40)]
    plane = _plane({})  # all default => hot
    out = plane.resolve(ToolContext(role="editor", role_rank=1), specs)
    assert len(out) == 40


def test_search_tool_always_kept():
    specs = [_spec("kit.search_tools"), _spec("pikkolo.x")]
    plane = _plane({"kit.search_tools": ToolVisibility(baseline="discoverable")})
    out = plane.resolve(ToolContext(role="editor", role_rank=1), specs)
    assert "kit.search_tools" in [s.name for s in out]
