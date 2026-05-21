"""Tool-phase handler — partition tool calls into auto-approve vs needs-user.

Inputs: ``ctx.metadata['pending_tool_calls']`` (list of dicts from streaming).
Outputs:
  - ``approved_tool_calls``, ``denied_tool_calls`` and ``unknown_tool_calls``
    populated.
  - Next phase: APPROVAL_WAIT if any needs-user, else TOOL_EXECUTING when
    there is any call to turn into a result, else TOOL_RESULTS.

Unknown tool names (no registered spec) are kept in their own bucket rather
than lumped with approval-denials: they need a distinct "unknown tool"
result so the model can self-correct, and TOOL_EXECUTING must run to build
that result. Routing straight to TOOL_RESULTS would skip result construction
entirely, leaving the model with silence.
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
    unknown: list[dict[str, Any]] = []

    specs_by_name = {s.name: s for s in registry.list_specs()}

    for call in pending:
        spec = specs_by_name.get(call["name"])
        if spec is None:
            unknown.append(call)
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
    ctx.metadata["unknown_tool_calls"] = unknown

    if needs_user:
        return Phase.APPROVAL_WAIT
    # TOOL_EXECUTING is the only phase that builds ToolResults — for denied
    # and unknown calls too, not just approved ones. Route there whenever
    # there is any call to turn into a result so the model always gets one.
    if auto_approve or auto_deny or unknown:
        return Phase.TOOL_EXECUTING
    return Phase.TOOL_RESULTS
