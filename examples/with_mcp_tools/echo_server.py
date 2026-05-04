"""Standalone MCP echo server used by the example.

Spawned as a subprocess by ``main.py``.
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
            name="reverse",
            description="Reverse the given string.",
            inputSchema={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name != "reverse":
        raise ValueError(f"unknown tool: {name}")
    return [TextContent(type="text", text=arguments["text"][::-1])]


if __name__ == "__main__":

    async def _run() -> None:
        async with stdio_server() as (reader, writer):
            await server.run(reader, writer, server.create_initialization_options())

    asyncio.run(_run())
