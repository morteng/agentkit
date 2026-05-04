"""Intent-gate handler — runs the IntentGate; reject -> ERRORED, else CONTEXT_BUILD."""

from typing import TYPE_CHECKING, Any

from agentkit.loop.context import TurnContext
from agentkit.loop.phase import Phase

if TYPE_CHECKING:
    from agentkit.guards.intent import IntentGate


async def handle_intent_gate(ctx: TurnContext, deps: dict[str, Any]) -> Phase:
    gate: IntentGate | None = deps.get("intent_gate")
    if gate is None:
        return Phase.CONTEXT_BUILD
    decision = await gate.evaluate(ctx)
    if decision.allow:
        return Phase.CONTEXT_BUILD
    ctx.metadata["intent_rejection_reason"] = decision.reason or "rejected"
    return Phase.ERRORED
