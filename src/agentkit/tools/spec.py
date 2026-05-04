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
