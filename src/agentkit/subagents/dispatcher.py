"""SubagentDispatcher — spawn a nested Loop and return the assistant's final text."""

import asyncio
from typing import Any

from agentkit._content import TextBlock
from agentkit._messages import MessageRole
from agentkit.errors import AgentkitError
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

        async for _ev in loop.run():
            pass

        # Extract the final assistant message text.
        for msg in reversed(child.history):
            if msg.role is MessageRole.ASSISTANT:
                texts = [b.text for b in msg.content if isinstance(b, TextBlock)]
                if texts:
                    return "\n".join(texts)
        return ""
