"""SubagentDispatcher — spawn a nested Loop and return the assistant's final text."""

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from agentkit._content import TextBlock
from agentkit._messages import MessageRole
from agentkit.errors import AgentkitError
from agentkit.events import TextDelta, ToolCallStarted
from agentkit.events.base import BaseEvent
from agentkit.loop.context import TurnContext
from agentkit.loop.handlers.approval_wait import handle_approval_wait
from agentkit.loop.handlers.context_build import handle_context_build
from agentkit.loop.handlers.finalize_check import handle_finalize_check
from agentkit.loop.handlers.intent_gate import handle_intent_gate
from agentkit.loop.handlers.memory_extract import handle_memory_extract
from agentkit.loop.handlers.streaming import handle_streaming
from agentkit.loop.handlers.tool_executing import handle_tool_executing
from agentkit.loop.handlers.tool_phase import handle_tool_phase
from agentkit.loop.handlers.tool_results import handle_tool_results
from agentkit.loop.orchestrator import Loop
from agentkit.loop.phase import Phase
from agentkit.subagents.isolation import fresh_child_context

# Buffer subagent text deltas this many characters before flushing as a
# parent-stream ToolCallProgress. Newlines also flush. Tuned so the parent
# UI sees a sentence-or-two cadence rather than a flood of single-char events.
_SUBAGENT_TEXT_FLUSH_AT = 80


async def _surface_subagent_events(parent: TurnContext, events: AsyncIterator[BaseEvent]) -> None:
    """Drain a child loop's events and surface key milestones as
    :class:`ToolCallProgress` on the parent's event queue.

    Two milestones reach the parent:
      * Child :class:`TextDelta` is debounced into chunks (flushed on newline
        or after ``_SUBAGENT_TEXT_FLUSH_AT`` characters) and emitted as
        ``"subagent: <text>"``.
      * Child :class:`ToolCallStarted` is emitted as
        ``"subagent calling <tool_name>"``.

    All other child events (PhaseChanged, ApprovalNeeded, MessageStarted,
    etc.) are consumed but not surfaced; they are internal to the child loop.
    """
    buffer: list[str] = []

    async def _flush() -> None:
        if not buffer:
            return
        text = "".join(buffer).strip()
        buffer.clear()
        if text:
            await parent.report_tool_progress(f"subagent: {text}")

    async for ev in events:
        if isinstance(ev, TextDelta):
            buffer.append(ev.delta)
            joined = "".join(buffer)
            if "\n" in joined or len(joined) >= _SUBAGENT_TEXT_FLUSH_AT:
                await _flush()
        elif isinstance(ev, ToolCallStarted):
            await _flush()
            await parent.report_tool_progress(f"subagent calling {ev.tool_name}")
    await _flush()


class SubagentDepthExceeded(AgentkitError):
    pass


class SubagentDispatcher:
    def __init__(self, *, deps: dict[str, Any], max_depth: int = 3) -> None:
        self._deps = deps
        self._max_depth = max_depth

    async def spawn(
        self,
        parent: TurnContext,
        *,
        prompt: str,
        tools: list[str],
        extra_context: dict[str, Any],
    ) -> str:
        depth = parent.metadata.get("subagent_depth", 0) + 1
        if depth > self._max_depth:
            raise SubagentDepthExceeded(f"max subagent depth {self._max_depth} exceeded")

        child = fresh_child_context(parent, prompt=prompt)
        child.metadata["subagent_depth"] = depth
        child.metadata["allowed_tools"] = tools
        child.metadata.update(extra_context)
        child.event_queue = asyncio.Queue()

        loop = Loop(
            ctx=child,
            handlers={
                Phase.INTENT_GATE: handle_intent_gate,
                Phase.CONTEXT_BUILD: handle_context_build,
                Phase.STREAMING: handle_streaming,
                Phase.TOOL_PHASE: handle_tool_phase,
                Phase.APPROVAL_WAIT: handle_approval_wait,
                Phase.TOOL_EXECUTING: handle_tool_executing,
                Phase.TOOL_RESULTS: handle_tool_results,
                Phase.FINALIZE_CHECK: handle_finalize_check,
                Phase.MEMORY_EXTRACT: handle_memory_extract,
            },
            deps=self._deps,
        )

        await _surface_subagent_events(parent, loop.run())

        # Extract the final assistant message text.
        for msg in reversed(child.history):
            if msg.role is MessageRole.ASSISTANT:
                texts = [b.text for b in msg.content if isinstance(b, TextBlock)]
                if texts:
                    return "\n".join(texts)
        return ""
