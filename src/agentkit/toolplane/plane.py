"""ToolPlane — per-turn catalog resolver.

Pure given a ToolContext. The streaming handler calls ``resolve`` as the
``tool_selector`` hook to filter ``registry.list_specs()`` each iteration.
``resolve`` also caches the discoverable tier so the ``search_tools``
builtin can match against it.
"""

from __future__ import annotations

import fnmatch
import logging
import re
from typing import TYPE_CHECKING, cast

from agentkit.toolplane.types import Tier, ToolContext, ToolDecision, ToolVisibility

if TYPE_CHECKING:
    from collections.abc import Callable

    from agentkit.tools.spec import ToolSpec

log = logging.getLogger(__name__)

_DEFAULT_VISIBILITY = ToolVisibility()  # baseline hot, no constraints
_TIER_RANK = {"hot": 0, "active": 1, "discoverable": 2, "hidden": 3}


def _bare(name: str) -> str:
    return name.split(".", 1)[-1]


def _matches_page(globs: list[str], page_path: str | None) -> bool:
    if not page_path or not globs:
        return False
    return any(fnmatch.fnmatch(page_path, g) for g in globs)


def _whole_word_hit(keywords: list[str], text: str | None) -> bool:
    if not keywords or not text:
        return False
    low = text.lower()
    return any(re.search(rf"\b{re.escape(kw.lower())}\b", low) for kw in keywords)


def _promoted_tier(vis: ToolVisibility, ctx: ToolContext) -> tuple[Tier, str] | None:
    """Return (tier, reason) if any declarative promotion rule fires, else None."""
    if _matches_page(vis.pages, ctx.page_path):
        return "active", f"page match {ctx.page_path}"
    if vis.features and (set(vis.features) & ctx.features):
        hit = sorted(set(vis.features) & ctx.features)
        return "active", f"feature match {hit}"
    if vis.entities and ctx.active_entity_kind in vis.entities:
        return "active", f"entity match {ctx.active_entity_kind}"
    if _whole_word_hit(vis.intent_keywords, ctx.recent_user_message):
        return "active", "intent keyword match"
    if vis.goals and ctx.active_goal_slug in vis.goals:
        return "active", f"goal match {ctx.active_goal_slug}"
    return None


class ToolPlane:
    HOT_CAP = 12
    ACTIVE_CAP = 30
    SEARCH_TOOL_NAME = "kit.search_tools"

    def __init__(
        self,
        *,
        visibility_of: Callable[[ToolSpec], ToolVisibility | None],
        context_of: Callable[[object], ToolContext],
        role_ranks: dict[str, int],
        rules: dict[str, Callable[[ToolContext], ToolDecision | None]] | None = None,
    ) -> None:
        self._visibility_of = visibility_of
        self._context_of = context_of
        self._role_ranks = role_ranks
        self._rules = rules or {}
        self._last_rationale: dict[str, ToolDecision] = {}
        self._last_discoverable: list[ToolSpec] = []

    @property
    def rationale(self) -> dict[str, ToolDecision]:
        return self._last_rationale

    @property
    def last_discoverable(self) -> list[ToolSpec]:
        return self._last_discoverable

    def resolve(self, turn_ctx: object, specs: list[ToolSpec]) -> list[ToolSpec]:
        """The ``tool_selector`` hook: returns the per-turn visible subset."""
        ctx = self._context_of(turn_ctx)
        decisions: dict[str, ToolDecision] = {}
        hot: list[ToolSpec] = []
        active: list[ToolSpec] = []
        discoverable: list[ToolSpec] = []
        for spec in specs:
            d = self._decide(spec, ctx)
            decisions[spec.name] = d
            if d.tier == "hot":
                hot.append(spec)
            elif d.tier == "active":
                active.append(spec)
            elif d.tier == "discoverable":
                discoverable.append(spec)
            # "hidden" dropped entirely
        self._last_rationale = decisions
        self._last_discoverable = discoverable

        # hot is intentionally uncapped: during the migration window every
        # tool is baseline=hot, and truncating would silently drop tools.
        # Only the active tier is capped. HOT_CAP is a soft signal.
        if len(hot) > self.HOT_CAP:
            log.debug(
                "toolplane: hot tier %d exceeds HOT_CAP %d (migration window)",
                len(hot),
                self.HOT_CAP,
            )
        result = hot + active[: self.ACTIVE_CAP]
        # Discovery escape hatch must always be reachable.
        if not any(s.name == self.SEARCH_TOOL_NAME for s in result):
            search = next((s for s in specs if s.name == self.SEARCH_TOOL_NAME), None)
            if search is not None:
                result.append(search)
        return result

    def hot_set(self, specs: list[ToolSpec], ctx: ToolContext) -> set[str]:
        """Names that resolve to the ``hot`` tier under ``ctx``.

        Pure: reuses ``_decide`` so it tracks the live resolution rules. The
        consumer passes a neutral context (no page/entity/discovery, top role)
        to derive its always-available core.
        """
        return {spec.name for spec in specs if self._decide(spec, ctx).tier == "hot"}

    def _decide(self, spec: ToolSpec, ctx: ToolContext) -> ToolDecision:
        vis = self._visibility_of(spec) or _DEFAULT_VISIBILITY
        name = spec.name
        bare = _bare(name)

        # 1. Hard gates — checked before any promotion.
        if vis.min_role is not None and ctx.role_rank < self._role_ranks.get(vis.min_role, 0):
            return ToolDecision("hidden", f"min_role={vis.min_role}, role={ctx.role}")
        if vis.mcp_clients is not None and (
            ctx.mcp_client is None or ctx.mcp_client not in vis.mcp_clients
        ):
            return ToolDecision("hidden", f"mcp_clients={vis.mcp_clients}, client={ctx.mcp_client}")

        # 2. Declarative promotion (first match wins), applied only when it
        # raises visibility — a promotion must never demote a more-visible
        # baseline (e.g. a hot tool with a pages= list stays hot off-cap).
        tier, reason = vis.baseline, f"baseline={vis.baseline}"
        promotion = _promoted_tier(vis, ctx)
        if promotion is not None:
            p_tier, p_reason = promotion
            if _TIER_RANK[p_tier] < _TIER_RANK[tier]:
                tier, reason = p_tier, p_reason

        # 3. Session-discovered tools promote to active (only if that raises visibility).
        if (bare in ctx.discovered_tools or name in ctx.discovered_tools) and _TIER_RANK[
            "active"
        ] < _TIER_RANK[tier]:
            tier, reason = "active", "discovered via search_tools"

        # 4. Pluggable rule can override.
        rule = self._rules.get(bare)
        if rule is not None:
            rd = rule(ctx)
            if rd is not None:
                tier, reason = rd.tier, rd.reason

        # 5. Explicit tier overrides (admin/test) win outright.
        ov = ctx.tier_overrides.get(name) or ctx.tier_overrides.get(bare)
        if ov is not None:
            tier = cast("Tier", ov)
            reason = f"override={ov}"

        return ToolDecision(tier, reason)
