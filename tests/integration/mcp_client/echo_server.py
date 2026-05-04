"""Minimal MCP-protocol-compatible echo server, used by stdio integration tests.

Speaks the official MCP JSON-RPC dialect via the ``mcp`` SDK so the test
exercises the real client/server handshake.
"""

import asyncio

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

server = Server("echo-server")


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="echo",
            description="Echo back the input.",
            inputSchema={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name != "echo":
        raise ValueError(f"unknown tool: {name}")
    return [TextContent(type="text", text=f"echo: {arguments['text']}")]


async def main() -> None:
    async with stdio_server() as (reader, writer):
        await server.run(reader, writer, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
