import pytest

from agentkit.errors import ToolError
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


def _spec(name: str, risk: RiskLevel = RiskLevel.READ) -> ToolSpec:
    return ToolSpec(
        name=name,
        description="d",
        parameters={"type": "object"},
        returns=None,
        risk=risk,
        idempotent=True,
        side_effects=SideEffects.NONE,
        requires_approval=ApprovalPolicy.BY_RISK,
        cache_ttl_seconds=None,
        timeout_seconds=10.0,
    )


@pytest.mark.asyncio
async def test_registry_invokes_registered_builtin():
    reg = ToolRegistry()

    async def handler(args, ctx):
        return ToolResult(
            call_id=ctx.call_id,
            status="ok",
            content=[ContentBlockOut(type="text", text=f"hi {args['n']}")],
            error=None,
            duration_ms=1,
            cached=False,
        )

    reg.register_builtin(_spec("kit.test"), handler)
    specs = reg.list_specs()
    assert any(s.name == "kit.test" for s in specs)

    res = await reg.invoke(ToolCall(id="c1", name="kit.test", arguments={"n": "x"}), ctx=_FakeCtx())  # type: ignore[arg-type]
    assert res.content[0].text == "hi x"


@pytest.mark.asyncio
async def test_registry_rejects_duplicate_registration():
    reg = ToolRegistry()

    async def h(args, ctx): ...

    reg.register_builtin(_spec("kit.test"), h)  # type: ignore[arg-type]
    with pytest.raises(ToolError):
        reg.register_builtin(_spec("kit.test"), h)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_registry_unknown_tool_raises():
    reg = ToolRegistry()
    with pytest.raises(ToolError):
        await reg.invoke(ToolCall(id="c", name="missing", arguments={}), ctx=_FakeCtx())  # type: ignore[arg-type]


class _FakeCtx:
    """Minimal stand-in for TurnContext (defined later)."""

    call_id = "c1"
