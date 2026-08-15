"""ToolPlane — per-turn catalog resolver.

Pure given a ToolContext. The streaming handler calls ``resolve`` as the
``tool_selector`` hook to filter ``registry.list_specs()`` each iteration.

Resolution results are **not** stored on the shared plane instance: two
sessions resolving concurrently would otherwise read each other's rationale
and discoverable tier. :meth:`ToolPlane.resolve_detailed` returns everything a
caller needs and keeps no state; :meth:`ToolPlane.resolve` additionally
publishes its :class:`Resolution` into a :class:`~contextvars.ContextVar` so
the legacy ``rationale`` / ``last_discoverable`` accessors (used by the
``search_tools`` builtin) stay correct — a ContextVar is per-task, so each
session's turn sees only its own resolution.
"""

from __future__ import annotations

import fnmatch
import logging
import re
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast
from weakref import WeakKeyDictionary

from agentkit.toolplane.types import Tier, ToolContext, ToolDecision, ToolVisibility

if TYPE_CHECKING:
    from collections.abc import Callable

    from agentkit.tools.spec import ToolSpec

log = logging.getLogger(__name__)

_DEFAULT_VISIBILITY = ToolVisibility()  # baseline hot, no constraints
_TIER_RANK = {"hot": 0, "active": 1, "discoverable": 2, "hidden": 3}


@dataclass(frozen=True)
class Resolution:
    """Everything one ``resolve`` produced, for one context. Immutable."""

    visible: list[ToolSpec] = field(default_factory=list)  # type: ignore[reportUnknownVariableType]
    """The subset advertised to the provider: hot + capped active (+ search)."""
    rationale: dict[str, ToolDecision] = field(default_factory=dict)  # type: ignore[reportUnknownVariableType]
    """Per-tool decision, including the tools that were dropped."""
    discoverable: list[ToolSpec] = field(default_factory=list)  # type: ignore[reportUnknownVariableType]
    """The discoverable tier — what ``kit.search_tools`` searches over."""


_EMPTY_RESOLUTION = Resolution()

# Keyed by the plane itself, so several planes can coexist in one context, and
# weakly, so a published resolution never keeps a dead plane (or its specs)
# alive. Rebound rather than mutated on every publish: a child task inherits
# the mapping and must not be able to write back into its parent's context.
_CURRENT: ContextVar[WeakKeyDictionary[ToolPlane, Resolution] | None] = ContextVar(
    "agentkit_toolplane_resolution", default=None
)


def tool_capability_satisfied(vis: ToolVisibility, capabilities: frozenset[str]) -> bool:
    """True when a tool is reachable under the given tenant capability set.

    An untagged tool (``vis.capability is None``) is always satisfied. A
    tagged tool requires its capability to be present. This is the single
    predicate both the visibility resolver (``_decide``) and the consumer's
    execution gate use, so the two layers cannot drift.
    """
    return vis.capability is None or vis.capability in capabilities


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

    # ---- Per-context resolution state --------------------------------------

    def current_resolution(self) -> Resolution:
        """The most recent :meth:`resolve` **in this execution context**.

        Empty when this context has not resolved yet. Never another session's
        resolution: the backing store is a ContextVar, and each turn runs in
        its own task.
        """
        published = _CURRENT.get()
        if published is None:
            return _EMPTY_RESOLUTION
        return published.get(self, _EMPTY_RESOLUTION)

    @property
    def rationale(self) -> dict[str, ToolDecision]:
        """Per-tool decisions from this context's most recent ``resolve``.

        Prefer :meth:`resolve_detailed`, which returns the rationale directly
        instead of reading it back out of context state.
        """
        return self.current_resolution().rationale

    @property
    def last_discoverable(self) -> list[ToolSpec]:
        """This context's discoverable tier — what ``kit.search_tools`` searches."""
        return self.current_resolution().discoverable

    def resolve(self, turn_ctx: object, specs: list[ToolSpec]) -> list[ToolSpec]:
        """The ``tool_selector`` hook: returns the per-turn visible subset."""
        resolution = self.resolve_detailed(turn_ctx, specs)
        # Copy-on-write: rebind the mapping rather than mutating the one this
        # context may share with a parent task.
        published: WeakKeyDictionary[ToolPlane, Resolution] = WeakKeyDictionary(
            _CURRENT.get() or {}
        )
        published[self] = resolution
        _CURRENT.set(published)
        return resolution.visible

    def resolve_detailed(self, turn_ctx: object, specs: list[ToolSpec]) -> Resolution:
        """Resolve ``specs`` for ``turn_ctx``. Pure — stores nothing anywhere."""
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
        return Resolution(visible=result, rationale=decisions, discoverable=discoverable)

    def context_for(self, turn_ctx: object) -> ToolContext:
        """Project the consumer's ToolContext out of a turn context."""
        return self._context_of(turn_ctx)

    def decide(self, spec: ToolSpec, ctx: ToolContext) -> ToolDecision:
        """The tier verdict for one tool under ``ctx``. Pure.

        Public so the execution-time gate can reach the *same* decision the
        advertisement path reached, rather than reimplementing it.
        """
        return self._decide(spec, ctx)

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
        if not tool_capability_satisfied(vis, ctx.capabilities):
            return ToolDecision("hidden", f"capability={vis.capability} not in tenant set")

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


class ToolPlaneAuthorizer:
    """Execution-time authorization gate backed by a :class:`ToolPlane`.

    Satisfies :class:`agentkit.tools.registry.ToolAuthorizer`. Filtering the
    advertised catalog is advisory — the model can still name a tool it was
    never shown, whether from an earlier turn, its system prompt, or an
    injected instruction. Installing this on the registry re-runs the very same
    ``_decide`` at invoke time, so a tool that resolves to ``hidden`` (min_role,
    mcp_clients, or an unsatisfied capability) is refused rather than executed.

    ``discoverable`` is intentionally allowed: those tools are legitimately
    reachable via ``kit.search_tools``, they are merely not advertised up front.
    Pass ``deny_tiers`` to tighten or loosen that.
    """

    def __init__(
        self,
        plane: ToolPlane,
        *,
        deny_tiers: frozenset[Tier] = frozenset({"hidden"}),
    ) -> None:
        self._plane = plane
        self._deny_tiers = deny_tiers

    def authorize(self, spec: ToolSpec, ctx: Any) -> str | None:
        decision = self._plane.decide(spec, self._plane.context_for(ctx))
        if decision.tier not in self._deny_tiers:
            return None
        return (
            f"denied: {spec.name} is not available in this context "
            f"({decision.reason}). It was not executed."
        )
