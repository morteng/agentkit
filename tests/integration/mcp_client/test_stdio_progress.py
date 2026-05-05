"""StdioMCPClient progress-callback wiring (slice 2 of F8).

Spins up the FastMCP-based ``progress_server`` fixture and verifies that
``call_tool(..., on_progress=...)`` receives one callback per progress
notification with the (message, progress, total) shape agentkit expects.
"""

import sys

import pytest

from agentkit.mcp_client.stdio import StdioMCPClient

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_stdio_progress_callback_receives_notifications() -> None:
    client = StdioMCPClient(
        name="progress",
        command=[sys.executable, "-m", "tests.integration.mcp_client.progress_server"],
    )
    await client.initialize()
    received: list[tuple[str, float | None, float | None]] = []

    async def on_progress(message: str, progress: float | None, total: float | None) -> None:
        received.append((message, progress, total))

    try:
        result = await client.call_tool("slow_count", {"steps": 3}, on_progress=on_progress)
    finally:
        await client.shutdown()

    assert result.status == "ok"
    # One callback per step. Some MCP versions deduplicate or batch — accept
    # >=1 with monotonically increasing progress.
    assert len(received) >= 1
    progresses = [p for _msg, p, _total in received if p is not None]
    assert progresses == sorted(progresses)
    assert max(progresses) <= 3.0
    # Total propagates through unchanged.
    totals = {t for _msg, _p, t in received}
    assert totals == {3.0}
    # Messages are non-empty strings.
    assert all(isinstance(m, str) and m for m, _p, _t in received)


@pytest.mark.asyncio
async def test_stdio_no_progress_callback_still_works() -> None:
    """Backward compat: omitting on_progress must behave exactly as before."""
    client = StdioMCPClient(
        name="progress",
        command=[sys.executable, "-m", "tests.integration.mcp_client.progress_server"],
    )
    await client.initialize()
    try:
        result = await client.call_tool("slow_count", {"steps": 2})
    finally:
        await client.shutdown()
    assert result.status == "ok"
