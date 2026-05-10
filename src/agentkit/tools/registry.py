"""ToolRegistry — single source of truth for what tools exist this session.

Three registration sources:
1. Built-in tools: registered with a Python handler. Use the ``kit.`` namespace.
2. In-process MCP clients: ``InProcessMCPClient`` instances; their tools come
   under ``<server_name>.<tool_name>``.
3. Stdio MCP clients: subprocess-backed MCP servers; same naming rule.
"""

from __future__ import annotations

import contextlib
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from agentkit.errors import ToolError as ToolErr
from agentkit.loop.context import TurnContext
from agentkit.tools.spec import ToolCall, ToolResult, ToolSpec

if TYPE_CHECKING:
    from agentkit.mcp_client.base import MCPClient


# A built-in handler signature: (arguments, ctx) -> ToolResult.
BuiltinHandler = Callable[[dict[str, Any], TurnContext], Awaitable[ToolResult]]


class ToolRegistry:
    def __init__(self) -> None:
        self._builtins: dict[str, tuple[ToolSpec, BuiltinHandler]] = {}
        self._mcp_servers: dict[str, MCPClient] = {}
        self._mcp_specs: dict[str, ToolSpec] = {}  # qualified_name -> spec
        self._failed_servers: dict[str, str] = {}  # name -> error message; tools unavailable

    # ---- Registration ------------------------------------------------------

    def register_builtin(self, spec: ToolSpec, handler: BuiltinHandler) -> None:
        if spec.name in self._builtins:
            raise ToolErr(f"duplicate builtin registration: {spec.name}")
        self._builtins[spec.name] = (spec, handler)

    def register_default_builtins(self) -> None:
        """Register every entry of :data:`DEFAULT_BUILTINS`.

        Convenience for the common case — equivalent to looping over
        ``DEFAULT_BUILTINS`` and calling :meth:`register_builtin` for each entry.
        """
        # Local import to avoid a module-load cycle (builtin handlers may import
        # from ``agentkit.tools`` indirectly).
        from agentkit.tools.builtin import DEFAULT_BUILTINS  # noqa: PLC0415

        for spec, handler in DEFAULT_BUILTINS:
            self.register_builtin(spec, handler)

    def register_mcp_server(self, name: str, client: MCPClient) -> None:
        """Register an MCP server. Call ``initialize_mcp_servers`` after to load tools."""
        if name in self._mcp_servers:
            raise ToolErr(f"duplicate MCP server registration: {name}")
        self._mcp_servers[name] = client

    async def initialize_mcp_servers(self) -> None:
        """Connect to every registered MCP server and import its tools.

        A server whose ``initialize()`` or ``list_tools()`` raises is marked as
        failed (see :attr:`failed_servers`) but does **not** abort the loop —
        other servers continue to load. The session can still run; the failed
        server's tools simply aren't available.
        """
        for server_name, client in self._mcp_servers.items():
            try:
                await client.initialize()
                specs = await client.list_tools()
            except Exception as exc:
                self._failed_servers[server_name] = f"{type(exc).__name__}: {exc}"
                # Best-effort cleanup — shutdown may itself raise on a partially
                # initialized client; we record the original error regardless.
                with contextlib.suppress(Exception):
                    await client.shutdown()
                continue
            for spec in specs:
                qualified = f"{server_name}.{spec.name}"
                if qualified in self._mcp_specs or qualified in self._builtins:
                    raise ToolErr(f"namespace collision: {qualified}")
                self._mcp_specs[qualified] = spec.model_copy(update={"name": qualified})

    @property
    def failed_servers(self) -> dict[str, str]:
        """Map of MCP server name to the error message from initialization.

        Empty when all registered servers initialized cleanly.
        """
        return dict(self._failed_servers)

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

    async def invoke(self, call: ToolCall, ctx: TurnContext) -> ToolResult:
        # Built-in and MCP handlers each construct their own ToolResult; the
        # MCP ``call_tool`` signature does not even carry the call_id. Stamp
        # ``call.id`` onto the returned result here so downstream serialisation
        # always emits a well-formed ``tool_call_id`` — Gemini rejects empty
        # strings outright ("Tool message must have either name or
        # tool_call_id"); other providers tolerate the gap silently.
        if call.name in self._builtins:
            _spec, handler = self._builtins[call.name]
            result = await handler(call.arguments, ctx)
            return result.model_copy(update={"call_id": call.id})
        if call.name in self._mcp_specs:
            server, _, bare = call.name.partition(".")
            client = self._mcp_servers.get(server)
            if client is None:
                raise ToolErr(f"server not found for tool: {call.name}")

            # Bridge MCP server-side progress notifications to user-facing
            # ToolCallProgress events. The dispatcher sets ctx.call_id to the
            # current call before invoking; report_tool_progress no-ops if
            # ctx has no event_queue (e.g., subagent-internal contexts).
            async def _on_progress(
                message: str, progress: float | None, total: float | None
            ) -> None:
                await ctx.report_tool_progress(
                    message, call_id=call.id, progress=progress, total=total
                )

            result = await client.call_tool(bare, call.arguments, on_progress=_on_progress)
            return result.model_copy(update={"call_id": call.id})
        raise ToolErr(f"unknown tool: {call.name}")
