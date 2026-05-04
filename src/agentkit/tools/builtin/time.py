"""kit.current_time — provide the current ISO-8601 timestamp.

Useful for grounding the LLM and for testability: tests inject a FixedClock.
"""

from typing import Any

from agentkit.loop.context import TurnContext
from agentkit.tools.spec import (
    ApprovalPolicy,
    ContentBlockOut,
    RiskLevel,
    SideEffects,
    ToolResult,
    ToolSpec,
)

CURRENT_TIME_SPEC = ToolSpec(
    name="kit.current_time",
    description="Return the current UTC time in ISO-8601 format.",
    parameters={"type": "object", "properties": {}, "required": []},
    returns=None,
    risk=RiskLevel.READ,
    idempotent=True,  # within a single turn, time barely moves
    side_effects=SideEffects.NONE,
    requires_approval=ApprovalPolicy.NEVER,
    cache_ttl_seconds=None,  # do not cache across turns
    timeout_seconds=2.0,
)


async def current_time_handler(args: dict[str, Any], ctx: TurnContext) -> ToolResult:
    return ToolResult(
        call_id=ctx.call_id,
        status="ok",
        content=[ContentBlockOut(type="text", text=ctx.clock.now().isoformat())],
        error=None,
        duration_ms=0,
        cached=False,
    )
