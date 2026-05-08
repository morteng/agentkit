"""Provider protocol and provider-agnostic request/event types.

Loop code branches on ``ProviderCapabilities`` only — never on provider name.
"""

from collections.abc import AsyncIterator
from decimal import Decimal
from typing import Annotated, Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from agentkit._messages import Message, Usage

# ---- Tool-choice -----------------------------------------------------------


ToolChoiceMode = Literal["auto", "none", "required"]


class NamedToolChoice(BaseModel):
    """Force the model to call a specific tool by name."""

    name: str


# ---- Request types ---------------------------------------------------------


class SystemBlock(BaseModel):
    """One block of the system prompt — separately cacheable."""

    text: str
    cache: bool = True  # Hint: include cache_control where supported


class ToolDefinition(BaseModel):
    """Provider-agnostic tool definition. Adapters translate to the SDK's format."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema


class ThinkingConfig(BaseModel):
    enabled: bool = False
    budget_tokens: int | None = None


class StopCondition(BaseModel):
    """Stop sequences or token-count caps. Adapters apply per-SDK."""

    stop_sequences: list[str] = Field(default_factory=list)  # type: ignore[reportUnknownVariableType]


class ProviderRequest(BaseModel):
    model: str
    system: list[SystemBlock] = Field(default_factory=list)  # type: ignore[reportUnknownVariableType]
    messages: list[Message] = Field(default_factory=list)  # type: ignore[reportUnknownVariableType]
    tools: list[ToolDefinition] = Field(default_factory=list)  # type: ignore[reportUnknownVariableType]
    max_tokens: int = 4096
    temperature: float | None = None
    thinking: ThinkingConfig | None = None
    stop_when: StopCondition | None = None
    tool_choice: ToolChoiceMode | NamedToolChoice = "auto"
    """Tool-call constraint for this request.

    - ``"auto"`` (default): provider picks whether to call a tool.
    - ``"none"``: model must not call any tool — text-only reply.
    - ``"required"``: model must call at least one tool.
    - ``NamedToolChoice(name=...)``: model must call the named tool.

    Adapters translate to provider-native shapes. Has no effect when ``tools``
    is empty (the provider has nothing to choose from).
    """
    metadata: dict[str, str] = Field(default_factory=dict)  # type: ignore[reportUnknownVariableType]


# ---- Event types -----------------------------------------------------------


class _ProviderEventBase(BaseModel):
    type: str

    model_config = {"frozen": True}


class _MessageStart(_ProviderEventBase):
    type: Literal["message_start"] = "message_start"  # type: ignore[reportIncompatibleVariableOverride]
    role: Literal["assistant"] = "assistant"


class _TextDelta(_ProviderEventBase):
    type: Literal["text_delta"] = "text_delta"  # type: ignore[reportIncompatibleVariableOverride]
    delta: str
    block_index: int = 0


class _ThinkingDelta(_ProviderEventBase):
    type: Literal["thinking_delta"] = "thinking_delta"  # type: ignore[reportIncompatibleVariableOverride]
    delta: str


class _ToolCallStart(_ProviderEventBase):
    type: Literal["tool_call_start"] = "tool_call_start"  # type: ignore[reportIncompatibleVariableOverride]
    call_id: str
    tool_name: str


class _ToolCallDelta(_ProviderEventBase):
    type: Literal["tool_call_delta"] = "tool_call_delta"  # type: ignore[reportIncompatibleVariableOverride]
    call_id: str
    arguments_delta: str  # provider sends partial JSON; adapter accumulates


class _ToolCallComplete(_ProviderEventBase):
    type: Literal["tool_call_complete"] = "tool_call_complete"  # type: ignore[reportIncompatibleVariableOverride]
    call_id: str
    tool_name: str
    arguments: dict[str, Any]


class _MessageComplete(_ProviderEventBase):
    type: Literal["message_complete"] = "message_complete"  # type: ignore[reportIncompatibleVariableOverride]
    finish_reason: Literal["end_turn", "tool_use", "max_tokens", "stop_sequence"]


class _UsageEvent(_ProviderEventBase):
    type: Literal["usage"] = "usage"  # type: ignore[reportIncompatibleVariableOverride]
    usage: Usage


class _ErrorEvent(_ProviderEventBase):
    type: Literal["error"] = "error"  # type: ignore[reportIncompatibleVariableOverride]
    code: str
    message: str
    recoverable: bool = False


ProviderEvent = Annotated[
    _MessageStart
    | _TextDelta
    | _ThinkingDelta
    | _ToolCallStart
    | _ToolCallDelta
    | _ToolCallComplete
    | _MessageComplete
    | _UsageEvent
    | _ErrorEvent,
    Field(discriminator="type"),
]


# Re-export the concrete classes under public names so adapters can construct them.
MessageStart = _MessageStart
TextDelta = _TextDelta
ThinkingDelta = _ThinkingDelta
ToolCallStart = _ToolCallStart
ToolCallDelta = _ToolCallDelta
ToolCallComplete = _ToolCallComplete
MessageComplete = _MessageComplete
UsageEvent = _UsageEvent
ErrorEvent = _ErrorEvent


# ---- Capabilities and Provider protocol -----------------------------------


class ProviderCapabilities(BaseModel):
    supports_tool_use: bool
    supports_parallel_tools: bool
    supports_prompt_caching: bool
    supports_vision: bool
    supports_thinking: bool
    max_context_tokens: int
    max_output_tokens: int


@runtime_checkable
class Provider(Protocol):
    name: str
    capabilities: ProviderCapabilities

    def stream(self, request: ProviderRequest) -> AsyncIterator[ProviderEvent]: ...

    def estimate_tokens(self, messages: list[Message]) -> int: ...

    def estimate_cost(self, usage: Usage) -> Decimal: ...
