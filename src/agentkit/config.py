"""AgentConfig — Pydantic-Settings configuration for the agent runtime.

Composed of nested config groups so consumers can override one slice at a time.
"""

from typing import Any

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LoopConfig(BaseModel):
    max_iterations: int = 10  # cap turns per top-level run() call
    max_finalize_retries: int = 2
    max_claim_corrections: int = 1
    streaming_chunk_timeout_seconds: float = 60.0
    builtin_tool_note_enabled: bool = False  # the kit.note opt-in


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

    model_config = SettingsConfigDict(
        env_prefix="AGENTKIT_",
        env_nested_delimiter="__",
        extra="ignore",
    )
