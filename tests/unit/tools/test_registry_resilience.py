"""F23: a failing MCP server must not abort registry initialization."""

from typing import Any

import pytest

from agentkit.tools.registry import ToolRegistry
from agentkit.tools.spec import (
    ApprovalPolicy,
    RiskLevel,
    SideEffects,
    ToolResult,
    ToolSpec,
)


class _FailingClient:
    """An MCPClient whose initialize() raises — same shape as a real one would."""

    def __init__(self, name: str, exc: Exception) -> None:
        self.name = name
        self._exc = exc
        self.shutdown_called = False

    async def initialize(self) -> None:
        raise self._exc

    async def list_tools(self) -> list[ToolSpec]:
        return []

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        raise RuntimeError("not initialized")

    async def shutdown(self) -> None:
        self.shutdown_called = True

    async def health_check(self) -> bool:
        return False


class _GoodClient:
    """An MCPClient that initializes fine and exposes one tool."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.shutdown_called = False

    async def initialize(self) -> None:
        return None

    async def list_tools(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                name="reverse",
                description="Reverse text",
                parameters={"type": "object"},
                returns=None,
                risk=RiskLevel.READ,
                idempotent=True,
                side_effects=SideEffects.NONE,
                requires_approval=ApprovalPolicy.NEVER,
                cache_ttl_seconds=None,
                timeout_seconds=10.0,
            )
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        return ToolResult(call_id="", status="ok", content=[])

    async def shutdown(self) -> None:
        self.shutdown_called = True

    async def health_check(self) -> bool:
        return True


@pytest.mark.asyncio
async def test_failing_mcp_server_records_error_without_aborting():
    reg = ToolRegistry()
    bad = _FailingClient("dead", FileNotFoundError("/no/such/binary"))
    good = _GoodClient("alive")
    reg.register_mcp_server("dead", bad)  # type: ignore[arg-type]
    reg.register_mcp_server("alive", good)  # type: ignore[arg-type]

    # Must not raise.
    await reg.initialize_mcp_servers()

    assert "dead" in reg.failed_servers
    assert "FileNotFoundError" in reg.failed_servers["dead"]
    assert "alive" not in reg.failed_servers
    # Bad server's tools are not in the registry; good server's tool is.
    names = [s.name for s in reg.list_specs()]
    assert "alive.reverse" in names
    assert not any(n.startswith("dead.") for n in names)
    # Best-effort cleanup of the failed client.
    assert bad.shutdown_called  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_failing_mcp_server_with_unrelated_shutdown_error_still_recorded():
    """If shutdown() of the failed client itself raises, the original error still wins."""

    class _NoisyShutdown(_FailingClient):
        async def shutdown(self) -> None:
            raise RuntimeError("shutdown also broken")

    reg = ToolRegistry()
    bad = _NoisyShutdown("dead", FileNotFoundError("nope"))
    reg.register_mcp_server("dead", bad)  # type: ignore[arg-type]

    await reg.initialize_mcp_servers()
    assert "FileNotFoundError" in reg.failed_servers["dead"]


def test_register_default_builtins_idempotent_in_intent():
    reg = ToolRegistry()
    reg.register_default_builtins()
    names = {s.name for s in reg.list_specs()}
    assert "kit.finalize" in names
    assert "kit.current_time" in names
