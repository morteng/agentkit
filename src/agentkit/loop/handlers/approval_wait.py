"""Approval-wait handler.

Persists ctx to a CheckpointStore (if available) and emits ApprovalNeeded
events for each pending user-approval call. Always transitions to TURN_ENDED
with reason AWAITING_APPROVAL — the orchestrator surfaces this and the consumer
calls AgentSession.resume_with_approval to continue.
"""

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from agentkit._ids import EventId, new_id
from agentkit.events import ApprovalNeeded
from agentkit.events.lifecycle import TurnEndReason
from agentkit.loop.context import TurnContext
from agentkit.loop.phase import Phase

if TYPE_CHECKING:
    from agentkit.tools.registry import ToolRegistry


async def handle_approval_wait(ctx: TurnContext, deps: dict[str, Any]) -> Phase:
    registry: ToolRegistry = deps["registry"]
    timeout_seconds: float = deps.get("approval_timeout_seconds", 24 * 60 * 60)
    queue = ctx.event_queue
    pending = ctx.metadata.get("pending_user_approvals", [])
    timeout_at = datetime.now(UTC) + timedelta(seconds=timeout_seconds)
    # Persist into metadata so resume_with_approval can enforce server-side.
    ctx.metadata["approval_timeout_at"] = timeout_at.isoformat()

    specs_by_name = {s.name: s for s in registry.list_specs()}
    sequence = ctx.metadata.get("event_sequence", 100)
    for call in pending:
        spec = specs_by_name.get(call["name"])
        risk_value = spec.risk.value if spec else "unknown"
        ev = ApprovalNeeded(
            event_id=new_id(EventId),
            session_id=ctx.session_id,
            turn_id=ctx.turn_id,
            ts=ctx.clock.now(),
            sequence=sequence,
            call_id=call["id"],
            tool_name=call["name"],
            arguments=call["arguments"],
            risk=risk_value,
            timeout_at=timeout_at,
        )
        sequence += 1
        if queue is not None:
            await queue.put(ev)
    ctx.metadata["event_sequence"] = sequence

    # Persist a checkpoint keyed by turn_id so the resume call can find it.
    checkpoint_store = deps.get("checkpoint_store")
    if checkpoint_store is not None:
        # Local imports to avoid module-level import cycle.
        from agentkit._ids import CheckpointId  # noqa: PLC0415
        from agentkit.loop.context import to_checkpoint_payload  # noqa: PLC0415

        payload = to_checkpoint_payload(ctx)
        await checkpoint_store.save(CheckpointId(f"approval:{ctx.turn_id}"), payload)
        ctx.metadata["checkpoint_id"] = f"approval:{ctx.turn_id}"

    # The Loop end_reason for this turn is overridden by the AgentSession when
    # it sees pending_user_approvals; here we transition to TURN_ENDED so the
    # orchestrator emits the suspend.
    ctx.metadata["suspend_reason"] = TurnEndReason.AWAITING_APPROVAL.value
    return Phase.TURN_ENDED
