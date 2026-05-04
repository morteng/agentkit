"""Memory-extract handler — fire-and-forget extraction post-turn.

v0.1: pass-through. Future versions can run a small LLM over the turn's
messages to surface durable facts and call ctx.memory_store.save(...).
"""

from typing import Any

from agentkit.loop.context import TurnContext
from agentkit.loop.phase import Phase


async def handle_memory_extract(ctx: TurnContext, deps: dict[str, Any]) -> Phase:
    return Phase.TURN_ENDED
