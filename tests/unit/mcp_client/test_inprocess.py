import pytest

from agentkit.mcp_client import InProcessMCPClient
from agentkit.tools.spec import (
    ApprovalPolicy,
    ContentBlockOut,
    RiskLevel,
    SideEffects,
    ToolResult,
    ToolSpec,
)


def _spec(name: str, parameters: dict | None = None) -> ToolSpec:
    return ToolSpec(
        name=name,
        description="d",
        parameters={"type": "object"} if parameters is None else parameters,
        returns=None,
        risk=RiskLevel.READ,
        idempotent=True,
        side_effects=SideEffects.NONE,
        requires_approval=ApprovalPolicy.BY_RISK,
        cache_ttl_seconds=None,
        timeout_seconds=10.0,
    )


# A schema with a required string field, mirroring the shape of real builtins
# like kit.memory.recall ({"key": {"type": "string"}}, required=["key"]).
_TYPED_SCHEMA = {
    "type": "object",
    "properties": {"x": {"type": "string"}},
    "required": ["x"],
}


async def _echo_handler(args):
    return ToolResult(
        call_id="c1",
        status="ok",
        content=[ContentBlockOut(type="text", text=f"got {args['x']}")],
        error=None,
        duration_ms=1,
        cached=False,
    )


@pytest.mark.asyncio
async def test_inprocess_client_lists_registered_tools():
    client = InProcessMCPClient(name="srv")

    async def h(args):
        return ToolResult(
            call_id="c1",
            status="ok",
            content=[ContentBlockOut(type="text", text="ok")],
            error=None,
            duration_ms=1,
            cached=False,
        )

    client.register_tool(_spec("hello"), h)
    await client.initialize()
    tools = await client.list_tools()
    assert [t.name for t in tools] == ["hello"]


@pytest.mark.asyncio
async def test_inprocess_call_tool_invokes_handler():
    client = InProcessMCPClient(name="srv")

    async def h(args):
        return ToolResult(
            call_id="c1",
            status="ok",
            content=[ContentBlockOut(type="text", text=f"got {args['x']}")],
            error=None,
            duration_ms=1,
            cached=False,
        )

    client.register_tool(_spec("hello"), h)
    await client.initialize()
    res = await client.call_tool("hello", {"x": 7})
    assert res.content[0].text == "got 7"


@pytest.mark.asyncio
async def test_inprocess_unknown_tool_raises():
    client = InProcessMCPClient(name="srv")
    await client.initialize()
    with pytest.raises(KeyError):
        await client.call_tool("nope", {})


@pytest.mark.asyncio
async def test_inprocess_accepts_on_progress_and_no_ops() -> None:
    """In-process tools have no transport for progress notifications. The
    parameter must be accepted (protocol uniformity with StdioMCPClient) and
    silently ignored — calling it would never happen because handlers run
    synchronously to completion."""
    client = InProcessMCPClient(name="srv")

    async def h(args):
        return ToolResult(
            call_id="c1",
            status="ok",
            content=[ContentBlockOut(type="text", text="ok")],
            error=None,
            duration_ms=1,
            cached=False,
        )

    client.register_tool(_spec("hello"), h)
    await client.initialize()

    callbacks: list[tuple[str, float | None, float | None]] = []

    async def on_progress(message: str, progress: float | None, total: float | None) -> None:
        callbacks.append((message, progress, total))

    res = await client.call_tool("hello", {}, on_progress=on_progress)
    assert res.status == "ok"
    assert callbacks == []  # no-op as documented


@pytest.mark.asyncio
async def test_inprocess_valid_args_pass_through_unchanged():
    client = InProcessMCPClient(name="srv")
    client.register_tool(_spec("hello", _TYPED_SCHEMA), _echo_handler)
    await client.initialize()
    res = await client.call_tool("hello", {"x": "7"})
    assert res.status == "ok"
    assert res.content[0].text == "got 7"


@pytest.mark.asyncio
async def test_inprocess_missing_required_field_returns_structured_error():
    client = InProcessMCPClient(name="srv")
    client.register_tool(_spec("hello", _TYPED_SCHEMA), _echo_handler)
    await client.initialize()

    res = await client.call_tool("hello", {})

    assert res.status == "error"
    assert res.error is not None
    assert res.error.code == "invalid_arguments"
    assert "x" in res.error.message
    assert "missing required field" in res.error.message


@pytest.mark.asyncio
async def test_inprocess_wrong_type_returns_structured_error():
    client = InProcessMCPClient(name="srv")
    client.register_tool(_spec("hello", _TYPED_SCHEMA), _echo_handler)
    await client.initialize()

    res = await client.call_tool("hello", {"x": 7})  # int, schema wants string

    assert res.status == "error"
    assert res.error is not None
    assert res.error.code == "invalid_arguments"
    assert "x" in res.error.message
    assert "string" in res.error.message


@pytest.mark.asyncio
async def test_inprocess_no_schema_tool_still_works():
    """A tool registered with an unconstrained/empty schema (no properties or
    required list — e.g. kit.current_time's ``{"type": "object", "properties":
    {}, "required": []}``) must not regress: any arguments dict is accepted
    and passed straight to the handler."""
    client = InProcessMCPClient(name="srv")

    async def h(args):
        return ToolResult(
            call_id="c1",
            status="ok",
            content=[ContentBlockOut(type="text", text="ok")],
            error=None,
            duration_ms=1,
            cached=False,
        )

    client.register_tool(_spec("hello"), h)  # default parameters={"type": "object"}
    await client.initialize()
    res = await client.call_tool("hello", {"anything": "goes", "x": 1})
    assert res.status == "ok"
    assert res.content[0].text == "ok"


@pytest.mark.asyncio
async def test_inprocess_validation_error_shape_matches_exception_path():
    """A validation failure and a handler exception must produce ToolResults
    with the same envelope shape (call_id, content, duration_ms, cached) —
    only ``error.code``/``error.message`` differ."""
    client = InProcessMCPClient(name="srv")

    async def boom(args):
        raise ValueError("boom")

    client.register_tool(_spec("typed", _TYPED_SCHEMA), _echo_handler)
    client.register_tool(_spec("boom"), boom)
    await client.initialize()

    validation_res = await client.call_tool("typed", {})
    exception_res = await client.call_tool("boom", {})

    assert validation_res.status == exception_res.status == "error"
    assert validation_res.call_id == exception_res.call_id == ""
    assert validation_res.content == exception_res.content == []
    assert validation_res.cached is exception_res.cached is False
    assert isinstance(validation_res.duration_ms, int)
    assert isinstance(exception_res.duration_ms, int)
    assert validation_res.error is not None
    assert exception_res.error is not None
    assert validation_res.error.code == "invalid_arguments"
    assert exception_res.error.code == "handler_exception"
