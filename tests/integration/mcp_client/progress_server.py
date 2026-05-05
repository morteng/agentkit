"""MCP server fixture that emits progress notifications for stdio tests.

Uses the FastMCP high-level helper so we can call ``Context.report_progress``
inside the tool body. Real MCP clients receive these as
``notifications/progress`` JSON-RPC messages and dispatch them to the
per-call ``progress_callback`` agentkit's :class:`StdioMCPClient` wires up.
"""

import asyncio

from mcp.server.fastmcp import Context, FastMCP

mcp = FastMCP("progress-server")


@mcp.tool()
async def slow_count(steps: int, ctx: Context) -> str:  # type: ignore[type-arg]
    """Count to ``steps``, reporting progress on each tick."""
    for i in range(1, steps + 1):
        # Yield to the event loop briefly so the notification can flush before
        # the tool returns; otherwise the result and progress notifications
        # can race in the same tick.
        await asyncio.sleep(0.01)
        await ctx.report_progress(progress=float(i), total=float(steps), message=f"step {i}")
    return f"counted to {steps}"


if __name__ == "__main__":
    mcp.run()
