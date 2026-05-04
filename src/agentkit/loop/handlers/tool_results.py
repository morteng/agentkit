"""Tool-results handler — fold results into context and decide next phase."""

from typing import TYPE_CHECKING, Any

from agentkit._content import ContentBlock, TextBlock, ToolResultBlock
from agentkit._ids import EventId, MessageId, new_id
from agentkit._messages import Message, MessageRole
from agentkit.events import ToolCallResult
from agentkit.loop.context import TurnContext
from agentkit.loop.phase import Phase

if TYPE_CHECKING:
    from agentkit.tools.spec import ToolResult


async def handle_tool_results(ctx: TurnContext, deps: dict[str, Any]) -> Phase:
    results: list[ToolResult] = ctx.metadata.get("tool_results", [])
    queue = ctx.event_queue

    # Append each result as a MessageRole.TOOL message and emit ToolCallResult events.
    if results:
        sequence = ctx.metadata.get("event_sequence", 200)
        for r in results:
            tr_blocks: list[ContentBlock] = [
                TextBlock(text=cb.text or "") for cb in r.content if cb.type == "text"
            ]
            msg_content: list[ContentBlock] = [
                ToolResultBlock(
                    tool_use_id=r.call_id,
                    content=tr_blocks,
                    is_error=(r.status != "ok"),
                )
            ]
            msg = Message(
                id=new_id(MessageId),
                session_id=ctx.session_id,
                role=MessageRole.TOOL,
                content=msg_content,
                created_at=ctx.clock.now(),
            )
            ctx.add_message(msg)

            if queue is not None:
                content_summary = "\n".join(b.text for b in tr_blocks if isinstance(b, TextBlock))[
                    :200
                ]
                ev = ToolCallResult(
                    event_id=new_id(EventId),
                    session_id=ctx.session_id,
                    turn_id=ctx.turn_id,
                    ts=ctx.clock.now(),
                    sequence=sequence,
                    call_id=r.call_id,
                    status=r.status,
                    content_summary=content_summary,
                    duration_ms=r.duration_ms,
                    cached=r.cached,
                )
                sequence += 1
                await queue.put(ev)
        ctx.metadata["event_sequence"] = sequence

    if ctx.finalize_called:
        return Phase.FINALIZE_CHECK
    # Iterate again — let the agent see results and continue.
    iterations = ctx.metadata.get("iterations", 0) + 1
    ctx.metadata["iterations"] = iterations
    max_iter = deps.get("max_iterations", 10)
    if iterations >= max_iter:
        ctx.metadata["max_iterations_hit"] = True
        return Phase.FINALIZE_CHECK
    return Phase.CONTEXT_BUILD
