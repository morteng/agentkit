"""AgentConfig — Pydantic-Settings configuration for the agent runtime.

Composed of nested config groups so consumers can override one slice at a time.
"""

from typing import Any

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LoopConfig(BaseModel):
    max_iterations: int = 10  # cap turns per top-level run() call
    max_finalize_retries: int = 2
    # How many times a turn that ends WITHOUT a finalize_response call is
    # re-prompted to finalize before the loop lets the turn end. Only
    # applies when a finalize validator is configured. One nudge is
    # enough in practice.
    max_missing_finalize_reprompts: int = 1
    # When a missing-finalize re-prompt fires (see above), constrain that
    # re-prompt turn to the finalize tool via tool_choice. Without this, a
    # model that already answered inline can burn a whole free-form turn
    # (thinking, re-narrating) before — or instead of — finalizing, holding
    # the consumer in a streaming state for minutes. Opt-in: the provider
    # must support named tool_choice, and the consumer must register a
    # finalize tool (bare name "finalize"/"finalize_response"). Default off
    # preserves the unconstrained re-prompt for consumers that have not
    # verified provider support.
    force_finalize_on_missing_reprompt: bool = False
    max_claim_corrections: int = 1
    streaming_chunk_timeout_seconds: float = 60.0
    builtin_tool_note_enabled: bool = False  # the kit.note opt-in
    max_subagent_depth: int = 3  # how deep nested kit.subagent.spawn can recurse
    # Force-end the turn after N back-to-back errors from the same tool name.
    max_consecutive_tool_errors: int = 3


class ToolDispatchConfig(BaseModel):
    max_parallel: int = 8


class EventsConfig(BaseModel):
    queue_size: int = 256
    publish_phase_changed: bool = True


class GuardConfig(BaseModel):
    """Guard implementations injected at construction time.

    Fields are typed ``Any`` to avoid circular imports. Validation happens at
    use-site via ``isinstance(..., Protocol)`` from runtime_checkable protocols.
    """

    intent: Any = None
    approval: Any = None
    finalize: Any = None
    success_claim: Any = None
    success_claim_enabled: bool = False
    approval_timeout_seconds: float = 24 * 60 * 60


class StoreBundle(BaseModel):
    session: Any = None  # SessionStore
    memory: Any = None  # MemoryStore
    checkpoint: Any = None  # CheckpointStore

    model_config = {"arbitrary_types_allowed": True}


class AgentConfig(BaseSettings):
    """Top-level config. Reads from env via ``AGENTKIT_*`` prefix when used as Settings."""

    loop: LoopConfig = Field(default_factory=LoopConfig)
    tool_dispatch: ToolDispatchConfig = Field(default_factory=ToolDispatchConfig)
    events: EventsConfig = Field(default_factory=EventsConfig)
    guards: GuardConfig = Field(default_factory=GuardConfig)
    stores: StoreBundle = Field(default_factory=StoreBundle)
    # Optional per-call provider override. When set, AgentSession ignores
    # the constructor's `provider` arg and calls selector(ctx) per iteration.
    # Used for tier routing, model A/B testing, or any per-iteration provider choice.
    # Typed Any (not Callable) to avoid circular imports with Provider — same
    # rationale as GuardConfig.intent / approval / finalize / success_claim.
    provider_selector: Any = None
    # Optional per-iteration model override. When set, the streaming handler
    # resolves ``model = selector(ctx)`` and threads it into MessageBuilder as
    # a per-build override, so ``ProviderRequest.model`` reflects the
    # current-iteration model rather than the session's constructor-time
    # ``model``. Orthogonal to ``provider_selector``: a consumer can swap
    # provider (for per-tier reasoning_effort) AND swap model (for the
    # actual model_id at the wire) using a shared tracker. Typed Any to
    # avoid circular imports with TurnContext — same rationale as
    # ``provider_selector``.
    model_selector: Any = None

    model_config = SettingsConfigDict(
        env_prefix="AGENTKIT_",
        env_nested_delimiter="__",
        extra="ignore",
    )
