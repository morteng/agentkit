"""IntentGate — pre-LLM checks (rate limit, length, blocklist)."""

import re
from collections import defaultdict, deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Protocol, runtime_checkable

from agentkit._content import TextBlock
from agentkit.loop.context import TurnContext


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


class InMemoryRateLimitCheck(IntentCheck):
    """Sliding-window rate limit. ``ctx.metadata['owner']`` is the bucket key.

    Production deployments should swap in a Redis-backed implementation.
    """

    def __init__(self, *, turns_per_minute: int) -> None:
        self._cap = turns_per_minute
        self._window: dict[str, deque[datetime]] = defaultdict(deque)

    async def evaluate(self, ctx: TurnContext) -> IntentDecision:
        owner = str(ctx.metadata.get("owner", "anon"))
        now = datetime.now(UTC)
        cutoff = now - timedelta(minutes=1)
        bucket = self._window[owner]
        while bucket and bucket[0] < cutoff:
            bucket.popleft()
        if len(bucket) >= self._cap:
            return IntentDecision(
                allow=False,
                reason=f"rate limit exceeded ({self._cap}/min)",
            )
        bucket.append(now)
        return IntentDecision(allow=True)
