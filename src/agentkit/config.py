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
    # Optional continuation evaluator. When the session has an active goal
    # (set via :meth:`AgentSession.set_goal`), this hook fires after each
    # terminal envelope to decide whether the goal is met. When unset, goals
    # are inert (the loop runs single-turn as if no goal existed). See
    # :mod:`agentkit.continuation` for the protocol.
    continuation_evaluator: Any = None

    model_config = SettingsConfigDict(
        env_prefix="AGENTKIT_",
        env_nested_delimiter="__",
        extra="ignore",
    )
