"""StdioMCPClient — talks to an MCP server subprocess over JSON-RPC stdio."""

import asyncio
import time
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from agentkit.mcp_client.base import MCPClient
from agentkit.tools.spec import (
    ApprovalPolicy,
    ContentBlockOut,
    RiskLevel,
    SideEffects,
    ToolError,
    ToolResult,
    ToolSpec,
)


class StdioMCPClient(MCPClient):
    """Spawn an MCP server subprocess and speak JSON-RPC over its stdio."""

    def __init__(
        self,
        name: str,
        command: list[str],
        *,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        startup_timeout_seconds: float = 10.0,
    ) -> None:
        if not command:
            raise ValueError("command must not be empty")
        self.name = name
        self._params = StdioServerParameters(
            command=command[0],
            args=command[1:],
            env=env,
            cwd=cwd,
        )
        self._startup_timeout = startup_timeout_seconds
        self._session: ClientSession | None = None
        self._stdio_ctx: Any = None
        self._client_ctx: Any = None

    async def initialize(self) -> None:
        self._stdio_ctx = stdio_client(self._params)
        read, write = await asyncio.wait_for(
            self._stdio_ctx.__aenter__(),
            timeout=self._startup_timeout,
        )
        self._client_ctx = ClientSession(read, write)
        session: ClientSession = await self._client_ctx.__aenter__()
        await session.initialize()
        self._session = session

    async def list_tools(self) -> list[ToolSpec]:
        if self._session is None:
            raise RuntimeError("call initialize() first")
        result = await self._session.list_tools()
        return [_mcp_tool_to_spec(t) for t in result.tools]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        if self._session is None:
            raise RuntimeError("call initialize() first")
        started = time.perf_counter()
        try:
            response = await self._session.call_tool(name, arguments)
        except Exception as exc:
            elapsed = int((time.perf_counter() - started) * 1000)
            return ToolResult(
                call_id="",
                status="error",
                content=[],
                error=ToolError(code="mcp_call_failed", message=str(exc)),
                duration_ms=elapsed,
                cached=False,
            )
        elapsed = int((time.perf_counter() - started) * 1000)
        if getattr(response, "isError", False):
            return ToolResult(
                call_id="",
                status="error",
                content=[],
                error=ToolError(code="mcp_tool_error", message=_text_of(response.content)),
                duration_ms=elapsed,
                cached=False,
            )
        return ToolResult(
            call_id="",
            status="ok",
            content=[_mcp_content_to_block(c) for c in response.content],
            error=None,
            duration_ms=elapsed,
            cached=False,
        )

    async def shutdown(self) -> None:
        if self._client_ctx is not None:
            await self._client_ctx.__aexit__(None, None, None)
            self._client_ctx = None
        if self._stdio_ctx is not None:
            await self._stdio_ctx.__aexit__(None, None, None)
            self._stdio_ctx = None
        self._session = None

    async def health_check(self) -> bool:
        if self._session is None:
            return False
        try:
            await self._session.list_tools()
            return True
        except Exception:
            return False


def _mcp_tool_to_spec(tool: Any) -> ToolSpec:
    """Translate the official ``mcp.types.Tool`` into our ``ToolSpec``.

    Risk-level and approval defaults are conservative: an MCP tool's spec
    doesn't carry agentkit-specific metadata, so we treat all subprocess MCP
    tools as ``LOW_WRITE`` by default. Consumers can wrap the registry to
    override per tool.
    """
    return ToolSpec(
        name=tool.name,
        description=tool.description or "",
        parameters=tool.inputSchema or {"type": "object"},
        returns=None,
        risk=RiskLevel.LOW_WRITE,
        idempotent=False,
        side_effects=SideEffects.EXTERNAL_REVERSIBLE,
        requires_approval=ApprovalPolicy.BY_RISK,
        cache_ttl_seconds=None,
        timeout_seconds=30.0,
    )


def _mcp_content_to_block(content: Any) -> ContentBlockOut:
    if getattr(content, "type", None) == "image":
        return ContentBlockOut(
            type="image",
            image_data=getattr(content, "data", None),
            media_type=getattr(content, "mimeType", None),
        )
    text = getattr(content, "text", "")
    return ContentBlockOut(type="text", text=text)


def _text_of(content_list: list[Any] | None) -> str:
    parts: list[str] = [getattr(c, "text", "") for c in (content_list or [])]
    return "\n".join(parts)
