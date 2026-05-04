import pytest

from agentkit.mcp_client import InProcessMCPClient
from agentkit.tools.spec import (
    ApprovalPolicy,
    ContentBlockOut,
    RiskLevel,
    SideEffects,
    ToolResult,
    ToolSpec,
)


def _spec(name: str) -> ToolSpec:
    return ToolSpec(
        name=name,
        description="d",
        parameters={"type": "object"},
        returns=None,
        risk=RiskLevel.READ,
        idempotent=True,
        side_effects=SideEffects.NONE,
        requires_approval=ApprovalPolicy.BY_RISK,
        cache_ttl_seconds=None,
        timeout_seconds=10.0,
    )


@pytest.mark.asyncio
async def test_inprocess_client_lists_registered_tools():
    client = InProcessMCPClient(name="srv")

    async def h(args):
        return ToolResult(
            call_id="c1",
            status="ok",
            content=[ContentBlockOut(type="text", text="ok")],
            error=None,
            duration_ms=1,
            cached=False,
        )

    client.register_tool(_spec("hello"), h)
    await client.initialize()
    tools = await client.list_tools()
    assert [t.name for t in tools] == ["hello"]


@pytest.mark.asyncio
async def test_inprocess_call_tool_invokes_handler():
    client = InProcessMCPClient(name="srv")

    async def h(args):
        return ToolResult(
            call_id="c1",
            status="ok",
            content=[ContentBlockOut(type="text", text=f"got {args['x']}")],
            error=None,
            duration_ms=1,
            cached=False,
        )

    client.register_tool(_spec("hello"), h)
    await client.initialize()
    res = await client.call_tool("hello", {"x": 7})
    assert res.content[0].text == "got 7"


@pytest.mark.asyncio
async def test_inprocess_unknown_tool_raises():
    client = InProcessMCPClient(name="srv")
    await client.initialize()
    with pytest.raises(KeyError):
        await client.call_tool("nope", {})
