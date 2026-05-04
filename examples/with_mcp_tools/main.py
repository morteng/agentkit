"""Demo: AgentSession + StdioMCPClient subprocess.

Spawns examples/with_mcp_tools/echo_server.py as a subprocess MCP server
and lets the agent call its ``reverse`` tool.
"""

import asyncio
import os
import sys

from agentkit import AgentConfig, AgentSession
from agentkit._ids import OwnerId
from agentkit.events import TextDelta, ToolCallResult, ToolCallStarted, TurnEnded
from agentkit.guards.approval import RiskBasedApprovalGate
from agentkit.mcp_client import StdioMCPClient
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

    echo = StdioMCPClient(
        name="echo",
        command=[sys.executable, "examples/with_mcp_tools/echo_server.py"],
    )
    registry.register_mcp_server("echo", echo)

    session = AgentSession(
        owner=OwnerId("user:demo"),
        config=config,
        provider=AnthropicProvider(api_key=api_key),
        registry=registry,
        model="claude-haiku-4-5-20251001",
        system_blocks=[
            SystemBlock(
                text=(
                    "You are an assistant. Use the echo.reverse tool to "
                    "reverse the user's text. Then call kit.finalize."
                )
            ),
        ],
    )

    async with session.run("Reverse 'hello world'.") as stream:
        async for event in stream:
            if isinstance(event, TextDelta):
                sys.stdout.write(event.delta)
                sys.stdout.flush()
            elif isinstance(event, ToolCallStarted):
                print(f"\n[tool call: {event.tool_name}({event.arguments})]")
            elif isinstance(event, ToolCallResult):
                print(f"[tool result: {event.status} ({event.duration_ms}ms)]")
            elif isinstance(event, TurnEnded):
                print(f"[turn ended: {event.reason.value}]")

    await session.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
