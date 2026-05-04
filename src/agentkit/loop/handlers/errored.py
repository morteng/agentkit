"""Errored handler — terminal phase.

The orchestrator does not call into terminal phases for handlers; this module
exists so callers can reference ``handle_errored`` symmetrically and future
versions can hook side effects (logging final state, alerting, etc.).
"""

from typing import Any

from agentkit.loop.context import TurnContext
from agentkit.loop.phase import Phase


async def handle_errored(ctx: TurnContext, deps: dict[str, Any]) -> Phase:
    return Phase.ERRORED  # never reached at runtime; satisfies signature
