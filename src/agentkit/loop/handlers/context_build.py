"""Context-build handler.

In v0.1 this is a thin pass-through: the Loop has already loaded session
history into ``ctx.history`` and registered tools into the registry. Future
versions could enrich context with RAG retrievals, automatic memory recall, etc.
"""

from typing import Any

from agentkit.loop.context import TurnContext
from agentkit.loop.phase import Phase


async def handle_context_build(ctx: TurnContext, deps: dict[str, Any]) -> Phase:
    return Phase.STREAMING
