"""MCPClient protocol — uniform interface across transports."""

from collections.abc import Awaitable, Callable
from typing import Any, Protocol, runtime_checkable

from agentkit.tools.spec import ToolResult, ToolSpec

ProgressCallback = Callable[[str, float | None, float | None], Awaitable[None]]
"""Per-call progress hook: ``(message, progress, total)``.

Argument order matches :meth:`TurnContext.report_tool_progress` rather than
the MCP SDK's ``(progress, total, message)`` so callers can forward to
``ctx.report_tool_progress`` without reordering.
"""


@runtime_checkable
class MCPClient(Protocol):
    name: str

    async def initialize(self) -> None: ...
    async def list_tools(self) -> list[ToolSpec]: ...
    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        on_progress: ProgressCallback | None = None,
    ) -> ToolResult: ...
    async def shutdown(self) -> None: ...
    async def health_check(self) -> bool: ...
