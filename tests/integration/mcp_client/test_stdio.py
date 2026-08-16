import sys

import pytest

from agentkit._content import Provenance
from agentkit.mcp_client.stdio import StdioMCPClient

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_stdio_client_initialise_list_call(tmp_path):
    client = StdioMCPClient(
        name="echo",
        command=[sys.executable, "-m", "tests.integration.mcp_client.echo_server"],
    )
    await client.initialize()
    try:
        tools = await client.list_tools()
        assert any(t.name == "echo" for t in tools)
        result = await client.call_tool("echo", {"text": "hi"})
        assert result.status == "ok"
        text = result.content[0].text
        assert text is not None
        assert "hi" in text
        # A real subprocess server is the "outside world" this client talks
        # to over JSON-RPC — its output must taint the turn that reads it.
        assert result.provenance is Provenance.UNTRUSTED
    finally:
        await client.shutdown()
