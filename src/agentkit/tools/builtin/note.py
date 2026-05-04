"""kit.note — opt-in scratchpad for the agent.

Some teams find a note tool helps multi-step reasoning; others find it
redundant noise. Off by default in AgentConfig; consumers can enable.
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

NOTE_SPEC = ToolSpec(
    name="kit.note",
    description=(
        "Append a private note to your reasoning scratchpad. The note is "
        "visible to you in subsequent steps but is not shown to the user."
    ),
    parameters={
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    },
    returns=None,
    risk=RiskLevel.READ,
    idempotent=False,
    side_effects=SideEffects.LOCAL,
    requires_approval=ApprovalPolicy.NEVER,
    cache_ttl_seconds=None,
    timeout_seconds=2.0,
)


async def note_handler(args: dict[str, Any], ctx: TurnContext) -> ToolResult:
    text = str(args.get("text", "")).strip()
    if text:
        ctx.scratchpad.append(text)
    return ToolResult(
        call_id=ctx.call_id,
        status="ok",
        content=[ContentBlockOut(type="text", text="noted")],
        error=None,
        duration_ms=0,
        cached=False,
    )
