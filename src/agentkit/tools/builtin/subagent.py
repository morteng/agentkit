"""kit.subagent.spawn — dispatch a nested Loop with isolated context.

The actual nested-loop machinery is implemented by ``subagents.dispatcher`` and
injected into the TurnContext by the parent Loop (``ctx.spawn_subagent``).
This builtin is a thin shim that calls that injected coroutine.
"""

from typing import Any

from agentkit.loop.context import TurnContext
from agentkit.tools.spec import (
    ApprovalPolicy,
    ContentBlockOut,
    RiskLevel,
    SideEffects,
    ToolError,
    ToolResult,
    ToolSpec,
)

SUBAGENT_SPAWN_SPEC = ToolSpec(
    name="kit.subagent.spawn",
    description=(
        "Spawn a focused subagent with a restricted set of tools to handle a "
        "subtask. The subagent's final message is returned as the tool result."
    ),
    parameters={
        "type": "object",
        "properties": {
            "prompt": {"type": "string"},
            "tools": {"type": "array", "items": {"type": "string"}},
            "context": {"type": "object"},
        },
        "required": ["prompt", "tools"],
    },
    returns=None,
    risk=RiskLevel.LOW_WRITE,
    idempotent=False,
    side_effects=SideEffects.LOCAL,
    requires_approval=ApprovalPolicy.NEVER,
    cache_ttl_seconds=None,
    timeout_seconds=120.0,
)


async def subagent_spawn_handler(args: dict[str, Any], ctx: TurnContext) -> ToolResult:
    spawn = ctx.spawn_subagent
    if spawn is None:
        return ToolResult(
            call_id=ctx.call_id,
            status="error",
            content=[],
            error=ToolError(code="subagent_not_configured", message="No subagent dispatcher."),
            duration_ms=0,
            cached=False,
        )
    summary = await spawn(
        str(args["prompt"]),
        list(args.get("tools", [])),
        dict(args.get("context", {})),
    )
    return ToolResult(
        call_id=ctx.call_id,
        status="ok",
        content=[ContentBlockOut(type="text", text=summary)],
        error=None,
        duration_ms=0,
        cached=False,
    )
