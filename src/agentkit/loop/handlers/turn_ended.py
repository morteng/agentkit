"""Turn-ended handler — terminal phase. Mirror of errored.py."""

from typing import Any

from agentkit.loop.context import TurnContext
from agentkit.loop.phase import Phase


async def handle_turn_ended(ctx: TurnContext, deps: dict[str, Any]) -> Phase:
    return Phase.TURN_ENDED
