from agentkit.tools.spec import (
    ApprovalPolicy,
    ContentBlockOut,
    RiskLevel,
    SideEffects,
    ToolCall,
    ToolResult,
    ToolSpec,
)


def test_tool_spec_defaults():
    spec = ToolSpec(
        name="test",
        description="d",
        parameters={"type": "object"},
        returns=None,
        risk=RiskLevel.READ,
        idempotent=True,
        side_effects=SideEffects.NONE,
        requires_approval=ApprovalPolicy.BY_RISK,
        cache_ttl_seconds=300,
        timeout_seconds=30.0,
    )
    assert spec.name == "test"


def test_tool_call_carries_arguments():
    call = ToolCall(id="c1", name="test", arguments={"x": 1})
    assert call.arguments == {"x": 1}


def test_tool_result_with_text_content():
    result = ToolResult(
        call_id="c1",
        status="ok",
        content=[ContentBlockOut(type="text", text="hello")],
        error=None,
        duration_ms=10,
        cached=False,
    )
    assert result.content[0].text == "hello"
