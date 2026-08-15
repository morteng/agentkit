"""kit.finalize — signal that the turn is complete.

The handler sets a flag on the TurnContext; the loop's finalize_check phase
then validates the claim before terminating.

**Two accepted argument shapes.** ``kit.finalize`` takes either a bare
``{"reason": "..."}`` or the full ``finalize_response`` envelope (``status``,
``intent_kind``, ``summary``, ``actions_performed``, …). Both are real: the
handler stores whatever it is given on ``ctx.finalize_args``, and
``loop.handlers.finalize_check`` hands that straight to the
:class:`~agentkit.guards.finalize.FinalizeValidator`, synthesizing
``{"reason": ...}`` only when nothing was recorded. The declared schema says so
— it previously advertised ``reason`` as the only property *and* as required,
which was never true of either the handler (it uses ``args.get``) or of any
caller in this repo. That drift was invisible until the registry started
validating arguments before dispatch; it is fixed here rather than worked
around, because the schema is what the model is told.

Nothing is required, for the same reason: an ``intent_kind="answer"`` envelope
is explicitly forbidden from carrying ``actions_performed``, so no single
required set covers both shapes. Semantic rules live in the finalize validator,
which can express "required *given* intent_kind" — a JSON Schema ``required``
list cannot.
"""

from typing import Any

from agentkit.loop.context import TurnContext
from agentkit.tools.builtin.finalize_response import FINALIZE_RESPONSE_SCHEMA
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
                    "One-sentence summary of what was completed. Surfaced on "
                    "TurnEnded.summary (the envelope's 'summary' is used when "
                    "'reason' is absent)."
                ),
            },
            **FINALIZE_RESPONSE_SCHEMA["properties"],
        },
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
    # Fall back to the envelope's ``summary``: an envelope-shaped call carries
    # the human-readable sentence there, and without this fallback every such
    # call ended the turn with ``TurnEnded.summary=None`` — the field the tool
    # description promises to fill.
    reason = str(args.get("reason") or args.get("summary") or "").strip()
    ctx.finalize_reason = reason or None
    ctx.finalize_args = dict(args)  # store full args for StructuralFinalizeValidator
    return ToolResult(
        call_id=ctx.call_id,
        status="ok",
        content=[ContentBlockOut(type="text", text="Finalize acknowledged.")],
        error=None,
        duration_ms=0,
        cached=False,
    )
