"""InProcessMCPClient — bypasses subprocess + JSON-RPC for tools in same process."""

import time
from collections.abc import Awaitable, Callable
from typing import Any

from agentkit._redaction import scrub_free_text
from agentkit.mcp_client.base import MCPClient, ProgressCallback
from agentkit.tools.spec import ToolError, ToolResult, ToolSpec
from agentkit.tools.validation import validate_arguments

InProcessHandler = Callable[[dict[str, Any]], Awaitable[ToolResult]]


class InProcessMCPClient(MCPClient):
    """Same-process MCP-shaped tool server.

    Handlers are async callables of ``(arguments) -> ToolResult``. Arguments
    are validated against the registered ``ToolSpec.parameters`` (JSON Schema)
    before dispatch — see ``agentkit.tools.validation`` — so a malformed call
    surfaces as a structured error result instead of an opaque handler-side
    exception. A real subprocess MCP server gets this for free from the MCP
    SDK; the in-process shortcut bypasses that transport, so it does its own
    check here. Otherwise the wire schema matches MCP sufficiently that
    swapping in a real MCP server (subprocess) is transparent to callers.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._handlers: dict[str, tuple[ToolSpec, InProcessHandler]] = {}
        self._initialized = False

    def register_tool(self, spec: ToolSpec, handler: InProcessHandler) -> None:
        if spec.name in self._handlers:
            raise ValueError(f"duplicate tool: {spec.name}")
        self._handlers[spec.name] = (spec, handler)

    async def initialize(self) -> None:
        self._initialized = True

    async def list_tools(self) -> list[ToolSpec]:
        if not self._initialized:
            raise RuntimeError("InProcessMCPClient not initialized")
        return [spec for spec, _ in self._handlers.values()]

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        on_progress: ProgressCallback | None = None,
    ) -> ToolResult:
        # In-process handlers have signature (arguments) -> ToolResult and run
        # synchronously to completion; there is no transport to deliver mid-
        # call progress over. Accept the parameter for protocol uniformity
        # (subprocess/stdio honors it) and silently ignore — handlers needing
        # real progress should be wired as builtins, which can call
        # ``ctx.report_tool_progress`` directly.
        _ = on_progress
        if name not in self._handlers:
            raise KeyError(name)
        _spec, handler = self._handlers[name]
        started = time.perf_counter()
        validation_error = validate_arguments(_spec, arguments)
        if validation_error is not None:
            elapsed = int((time.perf_counter() - started) * 1000)
            return ToolResult(
                call_id="",
                status="error",
                content=[],
                error=validation_error,
                duration_ms=elapsed,
                cached=False,
            )
        try:
            result = await handler(arguments)
        except Exception as exc:  # surface as ToolResult, not raise
            elapsed = int((time.perf_counter() - started) * 1000)
            return ToolResult(
                call_id="",
                status="error",
                content=[],
                # scrub_free_text, not str(exc): this message reaches an HTTP
                # surface and from there a browser. httpx embeds the full
                # request URL in its exception text, so an unscrubbed
                # str(exc) here has shipped an internal host, a port and an
                # api_key query param to a user's screen (observed
                # 2026-08-18 in a downstream consumer). Masking must happen where the
                # string is MADE — every consumer defending itself
                # separately is how one of them ends up not doing it.
                error=ToolError(code="handler_exception", message=scrub_free_text(str(exc))),
                duration_ms=elapsed,
                cached=False,
            )
        # Patch duration if handler didn't set it.
        if result.duration_ms == 0:
            result = result.model_copy(
                update={"duration_ms": int((time.perf_counter() - started) * 1000)}
            )
        return result

    async def shutdown(self) -> None:
        self._initialized = False

    async def health_check(self) -> bool:
        return self._initialized
