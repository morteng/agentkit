"""The on_iteration_start hook fires at the top of CONTEXT_BUILD.

This is the cooperative mid-run injection seam: the hook runs once before every
LLM call within a turn, and a message it appends to ``ctx.history`` is seen by
the next STREAMING build. See ``handlers/context_build.py``.
"""

import asyncio
from typing import Any

from agentkit.config import AgentConfig
from agentkit.loop.handlers.context_build import handle_context_build
from agentkit.loop.phase import Phase


def test_config_has_on_iteration_start_default_none():
    assert AgentConfig().on_iteration_start is None


class _FakeCtx:
    def __init__(self) -> None:
        self.history: list[Any] = []
        self.metadata: dict[str, Any] = {}

    def add_message(self, msg: Any) -> None:
        self.history.append(msg)


def test_handler_no_hook_is_passthrough():
    """Without a hook the handler is the original thin pass-through."""
    ctx = _FakeCtx()
    phase = asyncio.run(handle_context_build(ctx, {}))  # pyright: ignore[reportArgumentType]
    assert phase is Phase.STREAMING
    assert ctx.history == []


def test_handler_awaits_hook_with_ctx_and_history_append_survives():
    """The hook is awaited with ctx; a message it appends stays on history
    (which STREAMING rebuilds the next request from) and the phase still
    advances to STREAMING."""
    ctx = _FakeCtx()
    seen: dict[str, Any] = {}

    async def hook(c: Any) -> None:
        seen["ctx"] = c
        c.add_message("injected")

    phase = asyncio.run(handle_context_build(ctx, {"on_iteration_start": hook}))  # pyright: ignore[reportArgumentType]

    assert phase is Phase.STREAMING
    assert seen["ctx"] is ctx
    assert ctx.history == ["injected"]
