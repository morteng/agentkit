# with_mcp_tools — AgentSession with stdio MCP server

Spawns `echo_server.py` as a subprocess MCP server exposing `echo.reverse`,
and runs an AgentSession that uses the tool.

## Run

```bash
export ANTHROPIC_API_KEY=sk-...
uv run python examples/with_mcp_tools/main.py
```
