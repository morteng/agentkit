"""Generic, provider-agnostic Tool Plane types.

The resolver in ``plane.py`` interprets these declaratively. Field
*values* (page globs, feature flags, entity kinds, role names) are supplied
by the consuming application; agentkit assigns no meaning to the strings
beyond glob/membership matching and a consumer-provided role-rank map.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Tier = Literal["hot", "active", "discoverable", "hidden"]


@dataclass
class ToolVisibility:
    """Declarative visibility metadata attached to a tool by the consumer."""

    baseline: Tier = "hot"
    pages: list[str] = field(default_factory=list)  # type: ignore[reportUnknownVariableType]
    features: list[str] = field(default_factory=list)  # type: ignore[reportUnknownVariableType]
    entities: list[str] = field(default_factory=list)  # type: ignore[reportUnknownVariableType]
    intent_keywords: list[str] = field(default_factory=list)  # type: ignore[reportUnknownVariableType]
    goals: list[str] = field(default_factory=list)  # type: ignore[reportUnknownVariableType]
    min_role: str | None = None
    mcp_clients: list[str] | None = None
    capability: str | None = None  # inert seam: per-tenant capability gating (consumer-defined)


@dataclass
class ToolContext:
    """Per-turn signal bundle the consumer projects from its own context."""

    role: str
    role_rank: int
    mcp_client: str | None = None
    page_path: str | None = None
    active_entity_kind: str | None = None
    features: frozenset[str] = field(default_factory=frozenset)  # type: ignore[reportUnknownVariableType]
    recent_user_message: str | None = None
    active_goal_slug: str | None = None
    discovered_tools: frozenset[str] = field(default_factory=frozenset)  # type: ignore[reportUnknownVariableType]
    capabilities: frozenset[str] = field(default_factory=frozenset)  # type: ignore[reportUnknownVariableType]
    tier_overrides: dict[str, str] = field(default_factory=dict)  # type: ignore[reportUnknownVariableType]


@dataclass
class ToolDecision:
    """The resolver's verdict for one tool, with a human-readable reason."""

    tier: Tier
    reason: str
