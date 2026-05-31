"""In-memory BM25 over the discoverable tier + the search_tools builtin.

No external dependency: a compact BM25 Okapi over tool name + description.
The builtin matches the current ToolPlane's cached discoverable tier and
records hits via a consumer-supplied ``record`` callable (which the
consumer backs with session-scoped storage so discoveries persist).
"""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING

from agentkit.tools.spec import (
    ApprovalPolicy,
    ContentBlockOut,
    RiskLevel,
    SideEffects,
    ToolResult,
    ToolSpec,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

_TOKEN = re.compile(r"[a-z0-9]+")


def _tok(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def bm25_rank(
    query: str, docs: dict[str, str], *, limit: int = 8, k1: float = 1.5, b: float = 0.75
) -> list[tuple[str, float]]:
    """Rank ``docs`` (id -> text) against ``query`` by BM25. Returns top ``limit``."""
    q_terms = _tok(query)
    if not q_terms or not docs:
        return []
    tokenized = {doc_id: _tok(text) for doc_id, text in docs.items()}
    n = len(tokenized)
    avgdl = sum(len(t) for t in tokenized.values()) / n
    df: dict[str, int] = {}
    for term in set(q_terms):
        df[term] = sum(1 for toks in tokenized.values() if term in toks)
    scores: dict[str, float] = {}
    for doc_id, toks in tokenized.items():
        dl = len(toks)
        score = 0.0
        for term in q_terms:
            f = toks.count(term)
            if f == 0 or df.get(term, 0) == 0:
                continue
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            denom = f + k1 * (1 - b + b * dl / avgdl)
            score += idf * (f * (k1 + 1)) / denom
        scores[doc_id] = score
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return ranked[:limit]


_SEARCH_TOOLS_SPEC = ToolSpec(
    name="search_tools",  # registry namespaces to kit.search_tools
    description=(
        "Search for additional tools you might need when you don't see a "
        "capability in your current tools. Matches name, description, and "
        "keywords across the discoverable catalog; matched tools become "
        "available for the rest of this conversation."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What capability you need"},
            "limit": {"type": "integer", "description": "Max results", "default": 8},
        },
        "required": ["query"],
    },
    returns=None,
    risk=RiskLevel.READ,
    idempotent=True,
    side_effects=SideEffects.NONE,
    requires_approval=ApprovalPolicy.NEVER,
    cache_ttl_seconds=None,
    timeout_seconds=15.0,
)


def make_search_tools_builtin(
    plane: object,
    record: Callable[[object, list[str]], Awaitable[None]],
) -> tuple[ToolSpec, Callable[[dict[str, object], object], Awaitable[ToolResult]]]:
    """Build the (spec, handler) pair for the kit.search_tools builtin.

    ``plane`` is a ToolPlane whose ``last_discoverable`` is populated by the
    most recent ``resolve``. ``record(turn_ctx, names)`` is an async callable
    that persists discovered bare tool names so the resolver promotes them
    next iteration/turn.
    """

    async def handler(arguments: dict[str, object], ctx: object) -> ToolResult:
        query = str(arguments.get("query", "")).strip()
        raw_limit = arguments.get("limit", 8)
        limit = max(1, int(raw_limit)) if isinstance(raw_limit, (int, float)) else 8
        discoverable: list[ToolSpec] = list(getattr(plane, "last_discoverable", []))
        docs: dict[str, str] = {
            s.name: f"{s.name.split('.', 1)[-1]} {s.description}" for s in discoverable
        }
        ranked = bm25_rank(query, docs, limit=limit)
        # Filter zero-score results — they carry no relevance signal.
        ranked = [(doc_id, score) for doc_id, score in ranked if score > 0]
        if not ranked:
            return ToolResult(
                call_id="",
                status="ok",
                content=[ContentBlockOut(type="text", text="No additional tools matched.")],
            )
        by_name: dict[str, ToolSpec] = {s.name: s for s in discoverable}
        lines: list[str] = []
        bare_names: list[str] = []
        for full_name, _score in ranked:
            spec = by_name[full_name]
            bare = full_name.split(".", 1)[-1]
            bare_names.append(bare)
            lines.append(f"- {bare}: {spec.description}")
        await record(ctx, bare_names)
        text = "Matched tools (now available this conversation):\n" + "\n".join(lines)
        return ToolResult(
            call_id="",
            status="ok",
            content=[ContentBlockOut(type="text", text=text)],
        )

    return _SEARCH_TOOLS_SPEC, handler
