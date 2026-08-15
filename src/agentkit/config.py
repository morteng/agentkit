"""AgentConfig — Pydantic-Settings configuration for the agent runtime.

Composed of nested config groups so consumers can override one slice at a time.

Every field here is load-bearing: it changes runtime behaviour and a test proves
it. A knob that only looks like a knob is a lie told to the consumer, so a
setting the runtime cannot honour is deleted rather than kept for appearances.
"""

from typing import Any

from pydantic import BaseModel, Field, PrivateAttr
from pydantic_settings import BaseSettings, SettingsConfigDict

from agentkit.store.memory import MemoryScope


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
    # How many times a SuccessClaimGuard trip may interrupt the stream and
    # re-prompt the model within one turn (see loop.handlers.streaming). Each
    # trip costs a full extra model call, and the user has already seen the
    # interrupted text, so the budget is small. Once it is spent the guard is
    # disabled for the rest of the turn and the stream is allowed to finish:
    # a completed answer beats a turn that loops or dies. 0 disables the
    # correction entirely (the guard then only annotates ctx.metadata).
    max_claim_corrections: int = 1
    # Wall-clock budget for a single chunk from the provider stream. A provider
    # that accepts the request and then goes quiet (half-open socket, stuck
    # gateway) otherwise parks the turn forever with the consumer stuck in a
    # streaming state. On expiry the stream is closed and the turn takes the
    # normal recoverable-error path (retried when nothing was emitted yet).
    # Set 0 (or None) to wait indefinitely.
    streaming_chunk_timeout_seconds: float = 60.0
    max_subagent_depth: int = 3  # how deep nested kit.subagent.spawn can recurse
    # Force-end the turn after N back-to-back errors from the same tool name.
    max_consecutive_tool_errors: int = 3
    # Retry a stream that failed with a RECOVERABLE provider error (rate limit,
    # timeout, transient connection drop) up to this many times before
    # surfacing the error and ending the turn. The retry only fires when the
    # failed attempt had emitted no text or tool call yet (a clean early
    # failure), so it can never duplicate output the consumer already saw. This
    # is what keeps a long bulk turn alive across a brief provider blip instead
    # of aborting the whole worklist mid-flight. The budget is per stream
    # attempt: a successful stream resets it, so each iteration of a multi-step
    # turn gets a fresh allowance. Set 0 to disable (surface every error).
    max_stream_retries: int = 2
    # Base delay (seconds) for exponential backoff between stream retries:
    # ``delay = base * 2 ** (attempt - 1)``. Kept short — recoverable blips
    # (429s, momentary disconnects) typically clear within a second or two.
    stream_retry_base_delay_seconds: float = 0.5


class ToolDispatchConfig(BaseModel):
    max_parallel: int = 8


class EventsConfig(BaseModel):
    queue_size: int = 256
    # Emit PhaseChanged onto the event stream. Off keeps the phase machine's
    # internal chatter off the wire for consumers that only forward
    # user-facing events; ``ctx.phase_log`` is recorded either way, so the
    # per-phase timings survive for observability.
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
    # Per-principal turn rate limit, applied by the intent gate before any
    # model call. On by default: a runtime with privileged tools and no limit
    # lets one successful injection (or one runaway loop) spend without bound.
    # The ceiling is well above interactive human use — see
    # agentkit.guards.intent.DEFAULT_TURNS_PER_MINUTE. Set None to disable.
    rate_limit_turns_per_minute: int | None = 60

    # Cached because the limiter owns the sliding window: AgentSession rebuilds
    # its deps every turn, and a gate rebuilt per turn would forget every turn
    # it had just counted. Lives on the config object, which the consumer
    # constructs once and shares across sessions — so the window spans the
    # sessions of one principal, which is the point.
    _intent_gate: Any = PrivateAttr(default=None)

    def effective_intent_gate(self) -> Any:
        """The IntentGate the loop should run, or None when nothing is configured.

        Composes the built-in rate limiter with the consumer's own ``intent``
        gate (limiter first, so an over-quota turn is rejected before any
        custom check does work). Injecting ``intent`` therefore adds checks
        rather than silently replacing the rate limit — set
        ``rate_limit_turns_per_minute=None`` to opt out of it explicitly.
        """
        if self._intent_gate is None:
            # Local import: agentkit.guards.intent imports the loop context,
            # which must not depend on config. Same rationale as the ``Any``
            # field types above.
            from agentkit.guards.intent import (  # noqa: PLC0415
                DefaultIntentGate,
                InMemoryRateLimitCheck,
            )

            checks: list[Any] = []
            if self.rate_limit_turns_per_minute is not None:
                checks.append(
                    InMemoryRateLimitCheck(turns_per_minute=self.rate_limit_turns_per_minute)
                )
            if self.intent is not None:
                checks.append(self.intent)
            if not checks:
                return None
            self._intent_gate = DefaultIntentGate(checks=checks)
        return self._intent_gate


class StoreBundle(BaseModel):
    session: Any = None  # SessionStore
    memory: Any = None  # MemoryStore
    checkpoint: Any = None  # CheckpointStore
    # Scope handed to every MemoryStore call. Lives here, next to ``memory``,
    # because a store and its scope are only ever useful together: every
    # MemoryStore method takes a MemoryScope, and a scope without a store
    # addresses nothing. Consumers already reach for ``config.stores.memory``,
    # so the scope is discoverable in the same place.
    #
    # Unlike the store/guard fields above, this is typed concretely rather than
    # ``Any``: MemoryScope is a leaf pydantic model (stdlib + pydantic imports
    # only), so importing it here introduces no cycle — the ``Any`` idiom exists
    # for protocol types whose modules import back into the runtime.
    #
    # Left None, ``kit.memory.save``/``kit.memory.recall`` return the explicit
    # ``memory_not_configured`` error rather than silently no-op'ing.
    memory_scope: MemoryScope | None = None

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
    # Optional per-iteration tool-catalog filter. When set, the streaming
    # handler calls ``tool_selector(ctx, registry.list_specs())`` and sends
    # the returned subset to the provider, instead of the full catalog. Used
    # for progressive disclosure and Tool Plane routing. Typed Any to avoid
    # circular imports, same rationale as provider_selector and model_selector.
    tool_selector: Any = None
    # Where the runtime writes its audit records. Set it once here and every
    # tool dispatch and every approval verdict is recorded; leave it None and
    # AgentSession installs agentkit.audit.NullAuditSink, which writes nothing.
    # It is deliberately NOT a per-tool or per-client argument: that shape lets
    # one forgotten keyword leave the single most destructive tool in a runtime
    # unaudited while the harmless ones write rows. Wrap several sinks in
    # MultiAuditSink. Typed Any to avoid a circular import, same rationale as
    # the selectors above.
    audit_sink: Any = None
    # Optional async hook fired at the top of CONTEXT_BUILD — i.e. once before
    # every LLM call within a turn, not just at turn boundaries (the loop runs
    # TOOL_RESULTS -> CONTEXT_BUILD -> STREAMING each iteration). Signature
    # ``async def hook(ctx: TurnContext) -> None``; awaited, return value
    # ignored. Because STREAMING rebuilds the provider request from
    # ``ctx.history`` every iteration, a hook that appends a message to
    # ``ctx.history`` here has it seen by the very next model call — the
    # cooperative mid-run injection seam (drain a message inbox, add a recall,
    # nudge the turn) without a mid-token interrupt. The runner still persists
    # ``ctx.history`` at turn end, so it remains the single writer. Typed Any to
    # avoid circular imports with TurnContext, same rationale as the selectors.
    on_iteration_start: Any = None

    model_config = SettingsConfigDict(
        env_prefix="AGENTKIT_",
        env_nested_delimiter="__",
        extra="ignore",
    )
