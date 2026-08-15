"""IntentGate — pre-LLM checks (rate limit, length, blocklist).

The gate runs once per turn, before any model call, and a negative decision
routes the turn straight to ERRORED. It is the cheapest place in the runtime to
stop a turn: nothing has been spent and no tool has been reached.

:class:`InMemoryRateLimitCheck` is wired in by default — see
:meth:`agentkit.config.GuardConfig.effective_intent_gate`. An agent runtime with
privileged tools and no rate limit is how one successful prompt injection turns
into an unbounded sequence of turns; the default ceiling
(:data:`DEFAULT_TURNS_PER_MINUTE`) is set high enough that no interactive human
reaches it.
"""

import re
from collections import defaultdict, deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Protocol, runtime_checkable

from agentkit._content import TextBlock
from agentkit._logging import get_logger
from agentkit.loop.context import TurnContext

log = get_logger(__name__)

#: Turns per minute, per principal, allowed by the default gate. Deliberately
#: permissive: a human in a chat UI cannot reach it, a bulk automation run of
#: back-to-back turns stays under it, and a runaway loop is still bounded.
DEFAULT_TURNS_PER_MINUTE = 60


@dataclass(frozen=True)
class IntentDecision:
    allow: bool
    reason: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)  # type: ignore[reportUnknownVariableType]


@runtime_checkable
class IntentCheck(Protocol):
    async def evaluate(self, ctx: TurnContext) -> IntentDecision: ...


@runtime_checkable
class IntentGate(Protocol):
    async def evaluate(self, ctx: TurnContext) -> IntentDecision: ...


class DefaultIntentGate(IntentGate):
    """Compose checks; first negative wins."""

    def __init__(self, *, checks: Sequence[IntentCheck]) -> None:
        self._checks = list(checks)

    async def evaluate(self, ctx: TurnContext) -> IntentDecision:
        for check in self._checks:
            decision = await check.evaluate(ctx)
            if not decision.allow:
                return decision
        return IntentDecision(allow=True)


class MaxMessageLengthCheck(IntentCheck):
    """Reject if the user message has more than ``max_chars`` characters."""

    def __init__(self, *, max_chars: int) -> None:
        self._max = max_chars

    async def evaluate(self, ctx: TurnContext) -> IntentDecision:
        if not ctx.history:
            return IntentDecision(allow=True)
        last = ctx.history[-1]
        total = sum(len(b.text) for b in last.content if isinstance(b, TextBlock))
        if total > self._max:
            return IntentDecision(
                allow=False,
                reason=f"message exceeds max length ({total} > {self._max})",
            )
        return IntentDecision(allow=True)


class ContentBlocklistCheck(IntentCheck):
    """Reject if any blocklist regex matches the latest user message."""

    def __init__(self, *, patterns: Sequence[str]) -> None:
        self._patterns = [re.compile(p, re.IGNORECASE) for p in patterns]

    async def evaluate(self, ctx: TurnContext) -> IntentDecision:
        if not ctx.history:
            return IntentDecision(allow=True)
        last = ctx.history[-1]
        text = "\n".join(b.text for b in last.content if isinstance(b, TextBlock))
        for pat in self._patterns:
            if pat.search(text):
                return IntentDecision(
                    allow=False,
                    reason=f"content matches blocklist pattern: {pat.pattern}",
                )
        return IntentDecision(allow=True)


def verified_principal(ctx: TurnContext) -> str:
    """The rate-limit bucket key: who this turn is acting as.

    ``ctx.metadata["owner"]`` is the session's ``OwnerId``, stamped by
    :class:`~agentkit.session.AgentSession` from the constructor argument the
    *host* supplied — never by the model. It is re-stamped (not merged) onto a
    subagent child by the dispatcher, and ``owner`` is one of
    ``agentkit.subagents.isolation.RESERVED_CONTEXT_KEYS``, so a model-authored
    ``kit.subagent.spawn`` context cannot set it; it is also excluded from the
    checkpoint payload and rebuilt from the session on resume. That is what
    makes it *verified* — and what makes it the right key: a burst funnelled
    through spawned subagents counts against the principal that started it,
    which keying on ``session_id`` would not do.

    Falls back to the (equally runtime-owned) ``session_id`` when a context
    carries no owner, so an unowned turn gets its own bucket rather than
    sharing one global bucket with every other unowned session.
    """
    owner = ctx.metadata.get("owner")
    if owner:
        return f"owner:{owner}"
    return f"session:{ctx.session_id}"


class InMemoryRateLimitCheck(IntentCheck):
    """Sliding-window per-principal rate limit, in process memory.

    Keyed on :func:`verified_principal` — the runtime-stamped owner, not
    anything the model can influence. Pass ``principal_key`` to key on something
    else (a tenant id, an API key hash); the callable receives the
    :class:`~agentkit.loop.context.TurnContext` and must return a string that
    the model cannot choose.

    In-process state: each worker holds its own window, so N workers admit up to
    N times ``turns_per_minute``. That is the correct default for a self-hosted
    single-process deployment and a deliberate approximation elsewhere —
    multi-worker deployments should swap in a Redis-backed
    :class:`IntentCheck` with the same interface.
    """

    def __init__(
        self,
        *,
        turns_per_minute: int = DEFAULT_TURNS_PER_MINUTE,
        window_seconds: float = 60.0,
        principal_key: Callable[[TurnContext], str] | None = None,
    ) -> None:
        self._cap = turns_per_minute
        self._window_seconds = window_seconds
        self._key = principal_key or verified_principal
        self._window: dict[str, deque[datetime]] = defaultdict(deque)

    #: Prune stale buckets once the table grows past this. Without it the
    #: bucket table is an unbounded dict in a process that runs for months —
    #: the ``session:`` fallback key alone would add one entry per session.
    _PRUNE_ABOVE = 1024

    def _prune(self, cutoff: datetime) -> None:
        stale = [key for key, bucket in self._window.items() if not bucket or bucket[-1] < cutoff]
        for key in stale:
            del self._window[key]

    async def evaluate(self, ctx: TurnContext) -> IntentDecision:
        principal = self._key(ctx)
        now = datetime.now(UTC)
        cutoff = now - timedelta(seconds=self._window_seconds)
        if len(self._window) > self._PRUNE_ABOVE:
            self._prune(cutoff)
        bucket = self._window[principal]
        while bucket and bucket[0] < cutoff:
            bucket.popleft()
        if len(bucket) >= self._cap:
            # Logged, not just returned: a tripped limiter is either an abusive
            # client or a runaway agent, and both are worth seeing in the log.
            log.warning(
                "rate_limit_exceeded",
                limit=self._cap,
                window_seconds=self._window_seconds,
                session_id=str(ctx.session_id),
            )
            expires_at = bucket[0] + timedelta(seconds=self._window_seconds)
            retry_after = int((expires_at - now).total_seconds())
            return IntentDecision(
                allow=False,
                reason=(
                    f"rate limit exceeded ({self._cap} turns per "
                    f"{self._window_seconds:g}s); try again shortly"
                ),
                metadata={
                    "limit": self._cap,
                    "window_seconds": self._window_seconds,
                    "retry_after_seconds": max(retry_after, 1),
                },
            )
        bucket.append(now)
        return IntentDecision(allow=True)
