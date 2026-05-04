"""ToolRegistry — single source of truth for what tools exist this session.

Three registration sources:
1. Built-in tools: registered with a Python handler. Use the ``kit.`` namespace.
2. In-process MCP clients: ``InProcessMCPClient`` instances; their tools come
   under ``<server_name>.<tool_name>``.
3. Stdio MCP clients: subprocess-backed MCP servers; same naming rule.
"""

from collections.abc import Awaitable, Callable
from typing import Any, Protocol, runtime_checkable

from agentkit.errors import ToolError as ToolErr
from agentkit.tools.spec import ToolCall, ToolResult, ToolSpec


@runtime_checkable
class TurnContext(Protocol):
    """Minimal interface for loop context (full definition in agentkit.loop)."""

    call_id: str


@runtime_checkable
class MCPClient(Protocol):
    """Minimal interface for MCP client implementations (full definition in agentkit.mcp_client)."""

    async def initialize(self) -> None: ...
    async def list_tools(self) -> list[ToolSpec]: ...
    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult: ...
    async def shutdown(self) -> None: ...


# A built-in handler signature: (arguments, ctx) -> ToolResult.
BuiltinHandler = Callable[[dict[str, Any], Any], Awaitable[ToolResult]]


class ToolRegistry:
    def __init__(self) -> None:
        self._builtins: dict[str, tuple[ToolSpec, BuiltinHandler]] = {}
        self._mcp_servers: dict[str, MCPClient] = {}
        self._mcp_specs: dict[str, ToolSpec] = {}  # qualified_name -> spec

    # ---- Registration ------------------------------------------------------

    def register_builtin(self, spec: ToolSpec, handler: BuiltinHandler) -> None:
        if spec.name in self._builtins:
            raise ToolErr(f"duplicate builtin registration: {spec.name}")
        self._builtins[spec.name] = (spec, handler)

    def register_mcp_server(self, name: str, client: MCPClient) -> None:
        """Register an MCP server. Call ``initialize_mcp_servers`` after to load tools."""
        if name in self._mcp_servers:
            raise ToolErr(f"duplicate MCP server registration: {name}")
        self._mcp_servers[name] = client

    async def initialize_mcp_servers(self) -> None:
        """Connect to every registered MCP server and import its tools."""
        for server_name, client in self._mcp_servers.items():
            await client.initialize()
            specs = await client.list_tools()
            for spec in specs:
                qualified = f"{server_name}.{spec.name}"
                if qualified in self._mcp_specs or qualified in self._builtins:
                    raise ToolErr(f"namespace collision: {qualified}")
                self._mcp_specs[qualified] = spec.model_copy(update={"name": qualified})

    async def shutdown(self) -> None:
        for client in self._mcp_servers.values():
            await client.shutdown()

    # ---- Listing -----------------------------------------------------------

    def list_specs(self, *, available_to_provider: bool = True) -> list[ToolSpec]:
        # available_to_provider is forward API for risk-based filtering once the
        # ApprovalGate / TurnContext lands (Phase 7+). Today it is a no-op.
        _ = available_to_provider
        out = [spec for spec, _ in self._builtins.values()]
        out.extend(self._mcp_specs.values())
        return out

    # ---- Invocation --------------------------------------------------------

    async def invoke(self, call: ToolCall, ctx: Any) -> ToolResult:
        if call.name in self._builtins:
            _spec, handler = self._builtins[call.name]
            return await handler(call.arguments, ctx)
        if call.name in self._mcp_specs:
            server, _, bare = call.name.partition(".")
            client = self._mcp_servers.get(server)
            if client is None:
                raise ToolErr(f"server not found for tool: {call.name}")
            return await client.call_tool(bare, call.arguments)
        raise ToolErr(f"unknown tool: {call.name}")
