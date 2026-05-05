"""kit.request_approval — let the agent explicitly ask the user mid-flow.

⚠ Not enabled in DEFAULT_BUILTINS as of v0.1.1. The current implementation
appends to ``ctx.pending_approvals`` but **no orchestrator phase reads that
list** — calling this tool today is effectively a no-op for the user. Wiring
it through ``approval_wait`` requires an extra ``chosen_option`` parameter on
``AgentSession.resume_with_approval``; that work is tracked for a future
release. Consumers who want agent-initiated approvals today can register the
spec/handler manually and surface ``ctx.pending_approvals`` themselves.

Distinct from the per-tool ApprovalGate: this is a deliberate agent-initiated
request (e.g. "should I delete X or Y?").
"""

from dataclasses import dataclass
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


@dataclass
class PendingApproval:
    call_id: str
    prompt: str
    options: list[str]


REQUEST_APPROVAL_SPEC = ToolSpec(
    name="kit.request_approval",
    description="Ask the user to confirm or choose between options before continuing.",
    parameters={
        "type": "object",
        "properties": {
            "prompt": {"type": "string"},
            "options": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["prompt"],
    },
    returns=None,
    risk=RiskLevel.LOW_WRITE,
    idempotent=False,
    side_effects=SideEffects.LOCAL,
    requires_approval=ApprovalPolicy.NEVER,  # the request itself is not gated
    cache_ttl_seconds=None,
    timeout_seconds=5.0,
)


async def request_approval_handler(args: dict[str, Any], ctx: TurnContext) -> ToolResult:
    pa = PendingApproval(
        call_id=ctx.call_id,
        prompt=str(args["prompt"]),
        options=list(args.get("options", [])) or ["yes", "no"],
    )
    ctx.pending_approvals.append(pa)
    return ToolResult(
        call_id=ctx.call_id,
        status="ok",
        content=[
            ContentBlockOut(
                type="text",
                text="Approval requested. The user will be prompted before the next iteration.",
            )
        ],
        error=None,
        duration_ms=0,
        cached=False,
    )
