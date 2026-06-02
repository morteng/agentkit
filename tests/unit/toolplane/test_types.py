from agentkit.toolplane import ToolContext, ToolDecision, ToolVisibility


def test_visibility_defaults_to_hot_with_no_constraints():
    v = ToolVisibility()
    assert v.baseline == "hot"
    assert v.pages == [] and v.features == [] and v.entities == []
    assert v.intent_keywords == [] and v.goals == []
    assert v.min_role is None and v.mcp_clients is None
    assert v.capability is None


def test_visibility_carries_capability():
    v = ToolVisibility(baseline="discoverable", capability="restaurant_menu")
    assert v.capability == "restaurant_menu"


def test_context_minimal_construction():
    ctx = ToolContext(role="editor", role_rank=1)
    assert ctx.page_path is None
    assert ctx.features == frozenset()
    assert ctx.discovered_tools == frozenset()


def test_decision_carries_tier_and_reason():
    d = ToolDecision(tier="active", reason="page match /x")
    assert d.tier == "active"
    assert d.reason == "page match /x"
