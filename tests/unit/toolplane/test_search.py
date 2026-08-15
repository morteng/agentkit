from typing import cast

import pytest

from agentkit.toolplane import ToolPlane, make_search_tools_builtin
from agentkit.toolplane.search import bm25_rank
from agentkit.toolplane.types import ToolContext, ToolVisibility
from agentkit.tools.spec import ApprovalPolicy, RiskLevel, SideEffects, ToolSpec

ROLE_RANKS = {"viewer": 0, "editor": 1, "admin": 2, "superuser": 3}


def _as_ctx(turn_ctx: object) -> ToolContext:
    return cast("ToolContext", turn_ctx)


def _spec(name, desc):
    return ToolSpec(
        name=name,
        description=desc,
        parameters={},
        returns=None,
        risk=RiskLevel.READ,
        idempotent=True,
        side_effects=SideEffects.NONE,
        requires_approval=ApprovalPolicy.NEVER,
        cache_ttl_seconds=None,
        timeout_seconds=30.0,
    )


def test_bm25_ranks_relevant_doc_first():
    docs = {
        "a": "subtract one 3d shape from another boolean",
        "b": "translate content into another language",
        "c": "geocode an address to coordinates",
    }
    ranked = bm25_rank("3d csg subtract", docs, limit=2)
    assert ranked[0][0] == "a"
    assert len(ranked) == 2


@pytest.mark.asyncio
async def test_search_tools_builtin_matches_discoverable_and_records():
    specs = [
        _spec("acme.shape_subtract", "subtract one 3d shape from another"),
        _spec("acme.translate_content", "translate content to a language"),
        _spec("kit.search_tools", "search for tools"),
    ]
    plane = ToolPlane(
        visibility_of=lambda s: (
            ToolVisibility(baseline="discoverable") if s.name == "acme.shape_subtract" else None
        ),
        context_of=_as_ctx,
        role_ranks=ROLE_RANKS,
    )
    plane.resolve(ToolContext(role="editor", role_rank=1), specs)

    recorded: list[str] = []

    async def record(turn_ctx, names):
        recorded.extend(names)

    spec, handler = make_search_tools_builtin(plane, record)
    assert spec.name == "search_tools"  # bare; registry namespaces to kit.search_tools

    class _Ctx:
        pass

    result = await handler({"query": "3d csg", "limit": 5}, _Ctx())
    assert result.status == "ok"
    text = result.content[0].text
    assert text is not None
    # Advertise AND record the fully-qualified name — the only name the
    # registry routes. A bare "shape_subtract" here is unroutable.
    assert "acme.shape_subtract" in text
    assert "acme.shape_subtract" in recorded
    assert "shape_subtract" not in recorded  # never the bare form


@pytest.mark.asyncio
async def test_search_tools_advertises_qualified_names_verbatim():
    """Each advertised line must carry the full ``<server>.<tool>`` name — the
    exact string the registry routes — not the bare tool name."""
    specs = [
        _spec("acme.shape_subtract", "subtract one 3d shape from another"),
        _spec("kit.search_tools", "search for tools"),
    ]
    plane = ToolPlane(
        visibility_of=lambda s: (
            ToolVisibility(baseline="discoverable") if s.name == "acme.shape_subtract" else None
        ),
        context_of=_as_ctx,
        role_ranks=ROLE_RANKS,
    )
    plane.resolve(ToolContext(role="editor", role_rank=1), specs)

    async def record(turn_ctx, names):
        pass

    _, handler = make_search_tools_builtin(plane, record)

    class _Ctx:
        pass

    result = await handler({"query": "3d csg", "limit": 5}, _Ctx())
    text = result.content[0].text
    assert text is not None
    # The line is the fully-qualified name, verbatim — not "- shape_subtract:".
    assert "- acme.shape_subtract:" in text
    assert "- shape_subtract:" not in text


@pytest.mark.asyncio
async def test_search_tools_surfaced_name_is_invocable_verbatim():
    """A tool surfaced by search_tools must be invocable under the exact name
    shown. Regression guard for the prod bug: the advertised name was bare
    (``web_search``) while the registry only routed the qualified name
    (``acme.web_search``), so copying the advertised name hit unknown_tool.
    """
    from agentkit.tools.registry import ToolRegistry
    from agentkit.tools.spec import ContentBlockOut, ToolCall, ToolResult

    class _FakeMCPClient:
        name = "acme"

        async def initialize(self) -> None: ...
        async def list_tools(self):
            return [_spec("shape_subtract", "subtract one 3d shape from another")]

        async def call_tool(self, name, arguments, *, on_progress=None):
            return ToolResult(
                call_id="",
                status="ok",
                content=[ContentBlockOut(type="text", text="subtracted")],
            )

        async def shutdown(self) -> None: ...
        async def health_check(self) -> bool:
            return True

    reg = ToolRegistry()
    reg.register_mcp_server("acme", _FakeMCPClient())  # type: ignore[arg-type]
    await reg.initialize_mcp_servers()
    specs = reg.list_specs()  # yields the qualified "acme.shape_subtract"

    plane = ToolPlane(
        visibility_of=lambda s: (
            ToolVisibility(baseline="discoverable") if s.name == "acme.shape_subtract" else None
        ),
        context_of=_as_ctx,
        role_ranks=ROLE_RANKS,
    )
    plane.resolve(ToolContext(role="editor", role_rank=1), specs)

    async def record(turn_ctx, names):
        pass

    _, handler = make_search_tools_builtin(plane, record)

    class _Ctx:
        pass

    result = await handler({"query": "3d csg", "limit": 5}, _Ctx())
    text = result.content[0].text
    assert text is not None

    # Extract the name exactly as advertised to the model.
    shown = text.splitlines()[1].removeprefix("- ").split(":", 1)[0]
    assert shown == "acme.shape_subtract"

    # The advertised name routes cleanly — no unknown_tool.
    invoked = await reg.invoke(
        ToolCall(id="c1", name=shown, arguments={}),
        ctx=_FakeCtx(),  # type: ignore[arg-type]
    )
    assert invoked.status == "ok"
    assert invoked.error is None
    assert invoked.content[0].text == "subtracted"


class _FakeCtx:
    call_id = "c1"


@pytest.mark.asyncio
async def test_search_tools_negative_limit_does_not_drop_results():
    specs = [
        _spec("acme.shape_subtract", "subtract one 3d shape from another"),
        _spec("kit.search_tools", "search for tools"),
    ]
    plane = ToolPlane(
        visibility_of=lambda s: (
            ToolVisibility(baseline="discoverable") if s.name == "acme.shape_subtract" else None
        ),
        context_of=_as_ctx,
        role_ranks=ROLE_RANKS,
    )
    plane.resolve(ToolContext(role="editor", role_rank=1), specs)

    recorded: list[str] = []

    async def record(turn_ctx, names):
        recorded.extend(names)

    _, handler = make_search_tools_builtin(plane, record)

    class _Ctx:
        pass

    # A negative limit must not silently drop matched results via ranked[:-1] slicing.
    result = await handler({"query": "3d csg", "limit": -1}, _Ctx())
    assert result.status == "ok"
    text = result.content[0].text
    assert text is not None
    assert "shape_subtract" in text
