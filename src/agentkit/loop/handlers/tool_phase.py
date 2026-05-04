"""Tool-phase handler — partition tool calls into auto-approve vs needs-user.

Inputs: ``ctx.metadata['pending_tool_calls']`` (list of dicts from streaming).
Outputs:
  - ``approved_tool_calls`` and ``denied_tool_calls`` populated.
  - Next phase: APPROVAL_WAIT if any needs-user, else TOOL_EXECUTING (or
    TOOL_RESULTS if everything was auto-denied).
"""

from typing import TYPE_CHECKING, Any

from agentkit.guards.approval import ApprovalDecision
from agentkit.loop.context import TurnContext
from agentkit.loop.phase import Phase
from agentkit.tools.spec import ToolCall

if TYPE_CHECKING:
    from agentkit.guards.approval import ApprovalGate
    from agentkit.tools.registry import ToolRegistry


async def handle_tool_phase(ctx: TurnContext, deps: dict[str, Any]) -> Phase:
    registry: ToolRegistry = deps["registry"]
    gate: ApprovalGate = deps["approval_gate"]
    pending = ctx.metadata.get("pending_tool_calls", [])

    auto_approve: list[dict[str, Any]] = []
    needs_user: list[dict[str, Any]] = []
    auto_deny: list[dict[str, Any]] = []

    specs_by_name = {s.name: s for s in registry.list_specs()}

    for call in pending:
        spec = specs_by_name.get(call["name"])
        if spec is None:
            auto_deny.append(call)
            continue
        decision = await gate.decide(
            ToolCall(id=call["id"], name=call["name"], arguments=call["arguments"]),
            spec,
            ctx,
        )
        if decision is ApprovalDecision.AUTO_APPROVE:
            auto_approve.append(call)
        elif decision is ApprovalDecision.NEEDS_USER:
            needs_user.append(call)
        else:
            auto_deny.append(call)

    ctx.metadata["pending_user_approvals"] = needs_user
    ctx.metadata["approved_tool_calls"] = auto_approve
    ctx.metadata["denied_tool_calls"] = auto_deny

    if needs_user:
        return Phase.APPROVAL_WAIT
    if auto_approve:
        return Phase.TOOL_EXECUTING
    return Phase.TOOL_RESULTS
