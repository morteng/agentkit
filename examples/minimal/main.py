"""Minimal AgentSession demo.

Run: ``uv run python examples/minimal/main.py``

Requires ``ANTHROPIC_API_KEY`` set. Streams a single text-only turn from
Claude through the agent loop and prints deltas as they arrive.
"""

import asyncio
import os
import sys

from agentkit import AgentConfig, AgentSession
from agentkit._ids import OwnerId
from agentkit.events import TextDelta, TurnEnded
from agentkit.guards.approval import RiskBasedApprovalGate
from agentkit.providers.anthropic import AnthropicProvider
from agentkit.providers.base import SystemBlock
from agentkit.store.fakes import FakeCheckpointStore, FakeMemoryStore, FakeSessionStore
from agentkit.tools.builtin import DEFAULT_BUILTINS
from agentkit.tools.registry import ToolRegistry


async def main() -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ANTHROPIC_API_KEY is not set", file=sys.stderr)
        sys.exit(1)

    config = AgentConfig()
    config.guards.approval = RiskBasedApprovalGate()
    config.stores.session = FakeSessionStore()
    config.stores.memory = FakeMemoryStore()
    config.stores.checkpoint = FakeCheckpointStore()

    registry = ToolRegistry()
    for spec, handler in DEFAULT_BUILTINS:
        registry.register_builtin(spec, handler)

    session = AgentSession(
        owner=OwnerId("user:demo"),
        config=config,
        provider=AnthropicProvider(api_key=api_key),
        registry=registry,
        model="claude-haiku-4-5-20251001",
        system_blocks=[
            SystemBlock(text=("You are a helpful assistant. Use kit.finalize when you're done.")),
        ],
    )

    async with session.run("Say hello briefly.") as stream:
        async for event in stream:
            if isinstance(event, TextDelta):
                sys.stdout.write(event.delta)
                sys.stdout.flush()
            elif isinstance(event, TurnEnded):
                print()
                print(f"[turn ended: {event.reason.value}]")


if __name__ == "__main__":
    asyncio.run(main())
