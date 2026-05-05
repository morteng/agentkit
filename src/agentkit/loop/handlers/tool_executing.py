"""Tool-executing handler — invoke approved calls via the dispatcher."""

from typing import TYPE_CHECKING, Any

from agentkit.loop.context import TurnContext
from agentkit.loop.phase import Phase
from agentkit.tools.spec import ContentBlockOut, ToolCall, ToolResult

if TYPE_CHECKING:
    from agentkit.loop.tool_dispatcher import ToolDispatcher
    from agentkit.subagents.dispatcher import SubagentDispatcher


async def handle_tool_executing(ctx: TurnContext, deps: dict[str, Any]) -> Phase:
    dispatcher: ToolDispatcher = deps["dispatcher"]
    approved = ctx.metadata.get("approved_tool_calls", [])
    denied = ctx.metadata.get("denied_tool_calls", [])

    # Inject the spawn-subagent callable so kit.subagent.spawn can dispatch
    # nested loops. Skipped (left as None) when the session didn't wire one up.
    sub: SubagentDispatcher | None = deps.get("subagent_dispatcher")
    if sub is not None and ctx.spawn_subagent is None:

        async def _spawn(prompt: str, tools: list[str], extra: dict[str, Any]) -> str:
            return await sub.spawn(ctx, prompt=prompt, tools=tools, extra_context=extra)

        ctx.spawn_subagent = _spawn

    calls = [ToolCall(id=c["id"], name=c["name"], arguments=c["arguments"]) for c in approved]
    results: list[ToolResult] = []
    if calls:
        ctx.call_id = calls[0].id  # default; dispatcher overwrites per-call
        results = await dispatcher.run(calls, ctx)

    for c in denied:
        results.append(
            ToolResult(
                call_id=c["id"],
                status="denied",
                content=[ContentBlockOut(type="text", text="auto-denied by approval policy")],
                error=None,
                duration_ms=0,
                cached=False,
            )
        )

    ctx.metadata["tool_results"] = results
    return Phase.TOOL_RESULTS
