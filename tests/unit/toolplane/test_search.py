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
        "a": "subtract one 3d shape from another csg boolean",
        "b": "translate content into another language",
        "c": "geocode an address to coordinates",
    }
    ranked = bm25_rank("3d csg subtract", docs, limit=2)
    assert ranked[0][0] == "a"
    assert len(ranked) == 2


@pytest.mark.asyncio
async def test_search_tools_builtin_matches_discoverable_and_records():
    specs = [
        _spec("pikkolo.csg_subtract", "subtract one 3d shape from another csg"),
        _spec("pikkolo.translate_content", "translate content to a language"),
        _spec("kit.search_tools", "search for tools"),
    ]
    plane = ToolPlane(
        visibility_of=lambda s: (
            ToolVisibility(baseline="discoverable") if s.name == "pikkolo.csg_subtract" else None
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
    assert "csg_subtract" in text
    assert "csg_subtract" in recorded
