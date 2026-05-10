"""Finalize-check handler — validate the agent's finalize claim.

If finalize was not called this turn, treat as accepted (the conversation
naturally ended). If called, run the FinalizeValidator. Reject -> retry via
CONTEXT_BUILD with feedback injected; bounded by max_finalize_retries.
"""

from typing import TYPE_CHECKING, Any

from agentkit.loop.context import TurnContext
from agentkit.loop.phase import Phase
from agentkit.tools.spec import ToolCall

if TYPE_CHECKING:
    from agentkit.guards.finalize import FinalizeValidator


async def handle_finalize_check(ctx: TurnContext, deps: dict[str, Any]) -> Phase:
    validator: FinalizeValidator | None = deps.get("finalize_validator")
    if not ctx.finalize_called or validator is None:
        return Phase.MEMORY_EXTRACT

    finalize_call = ToolCall(
        id="finalize",
        name="kit.finalize",
        arguments=ctx.finalize_args
        if ctx.finalize_args is not None
        else {"reason": ctx.finalize_reason or ""},
    )
    verdict = await validator.validate(finalize_call, ctx)
    if verdict.accept:
        return Phase.MEMORY_EXTRACT

    retries = ctx.metadata.get("finalize_retries", 0)
    max_retries = deps.get("max_finalize_retries", 2)
    if retries >= max_retries:
        ctx.metadata["finalize_exhausted"] = True
        return Phase.MEMORY_EXTRACT
    ctx.metadata["finalize_retries"] = retries + 1
    ctx.metadata["finalize_correction"] = verdict.feedback or "Reconsider before finalizing."
    ctx.finalize_called = False
    ctx.finalize_reason = None
    return Phase.CONTEXT_BUILD
