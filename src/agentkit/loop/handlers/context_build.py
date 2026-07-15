"""Context-build handler.

Runs once before every LLM call within a turn (the loop cycles
TOOL_RESULTS -> CONTEXT_BUILD -> STREAMING each iteration), so it is the
natural place to enrich context per iteration: RAG retrievals, automatic
memory recall, or draining a cooperative mid-run message inbox.

The ``on_iteration_start`` config hook is awaited here. Because the STREAMING
handler rebuilds the provider request from ``ctx.history`` on the next step, a
hook that appends a message to ``ctx.history`` has it seen by the very next
model call — without interrupting a stream mid-token.
"""

from typing import Any

from agentkit.loop.context import TurnContext
from agentkit.loop.phase import Phase


async def handle_context_build(ctx: TurnContext, deps: dict[str, Any]) -> Phase:
    hook = deps.get("on_iteration_start")
    if hook is not None:
        await hook(ctx)
    return Phase.STREAMING
