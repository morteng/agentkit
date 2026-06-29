"""Canonical tool types — provider-agnostic."""

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


class RiskLevel(StrEnum):
    READ = "read"
    LOW_WRITE = "low_write"
    HIGH_WRITE = "high_write"
    DESTRUCTIVE = "destructive"


class SideEffects(StrEnum):
    NONE = "none"
    LOCAL = "local"
    EXTERNAL_REVERSIBLE = "external_reversible"
    EXTERNAL_IRREVERSIBLE = "external_irreversible"


class ApprovalPolicy(StrEnum):
    NEVER = "never"  # always auto-approve
    BY_RISK = "by_risk"  # let ApprovalGate decide based on risk
    ALWAYS = "always"  # always require user approval


class ContentBlockOut(BaseModel):
    """Content block returned by a tool (text or image)."""

    type: Literal["text", "image"]
    text: str | None = None
    image_url: str | None = None
    image_data: str | None = None
    media_type: str | None = None


class ToolError(BaseModel):
    """Structured error info on tool failure."""

    code: str
    message: str
    retryable: bool = False


class ToolSpec(BaseModel):
    """Provider-agnostic tool definition.

    The ``name`` is the auto-namespaced identifier (e.g. ``ampaera.devices.list``,
    ``kit.finalize``). Registry assigns the prefix; tool authors pass the bare name.
    """

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema
    returns: dict[str, Any] | None = None  # JSON Schema for result, optional
    risk: RiskLevel
    idempotent: bool
    side_effects: SideEffects
    requires_approval: ApprovalPolicy
    cache_ttl_seconds: int | None
    timeout_seconds: float


class ToolCall(BaseModel):
    id: str  # provider's tool_use_id
    name: str
    arguments: dict[str, Any]


class ToolResult(BaseModel):
    call_id: str
    status: Literal["ok", "error", "denied", "timeout", "cancelled"]
    content: list[ContentBlockOut] = Field(default_factory=list)  # type: ignore[reportUnknownVariableType]
    error: ToolError | None = None
    duration_ms: int = 0
    cached: bool = False


def unknown_tool_message(name: str) -> str:
    """The error text for a call to a name no registered tool matches.

    A dotted name (``content.get``, ``tasks.patch``) is almost always a
    scripting-namespace method the model reached for as a standalone tool. Naming
    that explicitly lets the model self-correct in one hop — call it inside the
    scripting tool, or use the matching flat tool if one exists — instead of
    ping-ponging on a bare "unknown tool". Plain names get the bare message.
    """
    if "." in name:
        return (
            f"unknown tool: {name}. Dotted names are scripting-namespace methods, "
            f"not standalone tools — call them inside the scripting tool "
            f"(await {name}(...)), or use the matching flat tool if one exists."
        )
    return f"unknown tool: {name}"
