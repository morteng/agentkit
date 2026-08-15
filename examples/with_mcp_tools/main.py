"""Demo: AgentSession + StdioMCPClient subprocess.

Spawns examples/with_mcp_tools/echo_server.py as a subprocess MCP server
and lets the agent call its ``reverse`` tool.

Also shows the ``classifier`` hook, which this demo now *needs*. MCP carries no
risk information, so an unclassified tool defaults to ``HIGH_WRITE`` with
``ApprovalPolicy.ALWAYS`` — without a classifier this example would stop and
wait for an approval nobody is there to grant. Vouching for a tool is the
operator's job, and it is deliberately explicit: you are asserting that you
know what ``echo.reverse`` does.
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
from agentkit.tools.spec import ApprovalPolicy, RiskLevel, SideEffects, ToolSpec


def classify_echo_tools(spec: ToolSpec) -> ToolSpec | None:
    """Vouch for the echo server's tools; leave anything else unclassified.

    Returning ``None`` means "I don't know this tool" — it keeps the
    fail-closed default (always ask the user) rather than waving it through.
    The returned spec must keep the same ``name``.
    """
    if spec.name.endswith(".reverse"):
        return spec.model_copy(
            update={
                "risk": RiskLevel.READ,
                "side_effects": SideEffects.NONE,
                "requires_approval": ApprovalPolicy.NEVER,
            }
        )
    return None


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
        classifier=classify_echo_tools,
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
