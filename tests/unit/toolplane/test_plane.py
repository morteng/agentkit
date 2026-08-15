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
    specs = [_spec("acme.search")]
    plane = _plane({})
    out = plane.resolve(ToolContext(role="editor", role_rank=1), specs)
    assert [s.name for s in out] == ["acme.search"]
    assert plane.rationale["acme.search"].tier == "hot"


def test_discoverable_baseline_is_dropped_from_array():
    specs = [_spec("acme.shape_subtract")]
    plane = _plane({"acme.shape_subtract": ToolVisibility(baseline="discoverable")})
    out = plane.resolve(ToolContext(role="editor", role_rank=1), specs)
    assert out == []
    assert plane.rationale["acme.shape_subtract"].tier == "discoverable"


def test_page_match_promotes_discoverable_to_active():
    specs = [_spec("acme.shape_subtract")]
    plane = _plane(
        {
            "acme.shape_subtract": ToolVisibility(
                baseline="discoverable", pages=["/dashboard/skins/*"]
            )
        }
    )
    ctx = ToolContext(role="editor", role_rank=1, page_path="/dashboard/skins/abc")
    out = plane.resolve(ctx, specs)
    assert [s.name for s in out] == ["acme.shape_subtract"]
    assert plane.rationale["acme.shape_subtract"].tier == "active"
    assert "page match" in plane.rationale["acme.shape_subtract"].reason


def test_min_role_hard_gate_hides_tool():
    specs = [_spec("acme.prune_facts")]
    plane = _plane({"acme.prune_facts": ToolVisibility(baseline="hot", min_role="admin")})
    ctx = ToolContext(role="editor", role_rank=1)
    out = plane.resolve(ctx, specs)
    assert out == []
    assert plane.rationale["acme.prune_facts"].tier == "hidden"


def test_intent_keyword_whole_word_match():
    specs = [_spec("acme.shape_subtract")]
    plane = _plane(
        {
            "acme.shape_subtract": ToolVisibility(
                baseline="discoverable", intent_keywords=["csg", "3d"]
            )
        }
    )
    hit = ToolContext(role="editor", role_rank=1, recent_user_message="make a 3d model please")
    miss = ToolContext(role="editor", role_rank=1, recent_user_message="threendimensional")
    assert [s.name for s in plane.resolve(hit, specs)] == ["acme.shape_subtract"]
    assert plane.resolve(miss, specs) == []


def test_discovered_set_promotes_to_active():
    specs = [_spec("acme.shape_subtract")]
    plane = _plane({"acme.shape_subtract": ToolVisibility(baseline="discoverable")})
    ctx = ToolContext(role="editor", role_rank=1, discovered_tools=frozenset({"shape_subtract"}))
    assert [s.name for s in plane.resolve(ctx, specs)] == ["acme.shape_subtract"]


def test_pluggable_rule_overrides_declarative():
    specs = [_spec("acme.prune_facts")]
    rules = {"prune_facts": lambda ctx: ToolDecision("hidden", "rule: too few facts")}
    plane = _plane({"acme.prune_facts": ToolVisibility(baseline="hot")}, rules=rules)
    out = plane.resolve(ToolContext(role="admin", role_rank=2), specs)
    assert out == []
    assert "rule:" in plane.rationale["acme.prune_facts"].reason


def test_active_cap_enforced():
    specs = [_spec(f"acme.t{i}") for i in range(40)]
    vis = {s.name: ToolVisibility(baseline="active") for s in specs}
    plane = _plane(vis)
    out = plane.resolve(ToolContext(role="editor", role_rank=1), specs)
    assert len(out) <= ToolPlane.ACTIVE_CAP


def test_hot_tier_not_truncated():
    # 40 hot tools must all survive — hot is uncapped during migration.
    specs = [_spec(f"acme.h{i}") for i in range(40)]
    plane = _plane({})  # all default => hot
    out = plane.resolve(ToolContext(role="editor", role_rank=1), specs)
    assert len(out) == 40


def test_search_tool_always_kept():
    specs = [_spec("kit.search_tools"), _spec("acme.x")]
    plane = _plane({"kit.search_tools": ToolVisibility(baseline="discoverable")})
    out = plane.resolve(ToolContext(role="editor", role_rank=1), specs)
    assert "kit.search_tools" in [s.name for s in out]


def test_hot_baseline_not_demoted_by_page_match():
    specs = [_spec("acme.always")]
    plane = _plane({"acme.always": ToolVisibility(baseline="hot", pages=["/dashboard/x/*"])})
    ctx = ToolContext(role="editor", role_rank=1, page_path="/dashboard/x/1")
    out = plane.resolve(ctx, specs)
    assert [s.name for s in out] == ["acme.always"]
    assert plane.rationale["acme.always"].tier == "hot"  # stayed hot, not demoted to active


def test_discovered_does_not_demote_hot_tool():
    specs = [_spec("acme.core")]
    plane = _plane({"acme.core": ToolVisibility(baseline="hot")})
    ctx = ToolContext(role="editor", role_rank=1, discovered_tools=frozenset({"core"}))
    plane.resolve(ctx, specs)
    assert plane.rationale["acme.core"].tier == "hot"


def test_mcp_client_hard_gate_hides_and_allows():
    specs = [_spec("acme.platform_health")]
    plane = _plane({"acme.platform_health": ToolVisibility(baseline="hot", mcp_clients=["cursor"])})
    # client not in allowlist -> hidden
    out = plane.resolve(ToolContext(role="editor", role_rank=1, mcp_client="vscode"), specs)
    assert out == []
    assert plane.rationale["acme.platform_health"].tier == "hidden"
    # no client at all -> hidden
    out_none = plane.resolve(ToolContext(role="editor", role_rank=1, mcp_client=None), specs)
    assert out_none == []
    # client in allowlist -> visible
    out_ok = plane.resolve(ToolContext(role="editor", role_rank=1, mcp_client="cursor"), specs)
    assert [s.name for s in out_ok] == ["acme.platform_health"]
