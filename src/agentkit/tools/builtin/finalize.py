"""kit.finalize — signal that the turn is complete.

The handler sets a flag on the TurnContext; the loop's finalize_check phase
then validates the claim before terminating.
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

FINALIZE_SPEC = ToolSpec(
    name="kit.finalize",
    description=(
        "Signal that you have completed the user's request. "
        "Provide a one-sentence summary of what you accomplished — it "
        "surfaces to the consumer on TurnEnded.summary, so write it for a "
        "human reading an audit log, not for yourself. "
        "Only call this once you have actually invoked the tools needed to "
        "carry out the user's request — do NOT call it just because you have "
        "produced a written response."
    ),
    parameters={
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "description": (
                    "One-sentence summary of what was completed. Surfaced on TurnEnded.summary."
                ),
            },
        },
        "required": ["reason"],
    },
    returns=None,
    risk=RiskLevel.READ,
    idempotent=True,
    side_effects=SideEffects.LOCAL,
    requires_approval=ApprovalPolicy.NEVER,
    cache_ttl_seconds=None,
    timeout_seconds=5.0,
)


async def finalize_handler(args: dict[str, Any], ctx: TurnContext) -> ToolResult:
    ctx.finalize_called = True
    ctx.finalize_reason = str(args.get("reason", "")).strip() or None
    return ToolResult(
        call_id=ctx.call_id,
        status="ok",
        content=[ContentBlockOut(type="text", text="Finalize acknowledged.")],
        error=None,
        duration_ms=0,
        cached=False,
    )
