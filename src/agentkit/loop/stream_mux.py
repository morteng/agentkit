"""StreamMux — translate ProviderEvents into user-facing Events.

Owns the assistant ``message_id`` used by streaming events. Sequence numbers
are allocated via :meth:`TurnContext.next_sequence` so the whole turn shares
a single monotonic counter across orchestrator, mux, and tool-progress
emitters. Deferred ``ToolCallStarted`` (with arguments) emits when the
provider sends ``tool_call_complete`` so the consumer sees a single event
with the full args, not a deluge of partial-JSON deltas.
"""

from collections.abc import AsyncIterator
from typing import Any

from agentkit._ids import EventId, MessageId, new_id
from agentkit._logging import get_logger
from agentkit.events import (
    ErrorCode,
    Errored,
    MessageCompleted,
    MessageStarted,
    ProviderActivity,
    TextDelta,
    ThinkingDelta,
    ToolCallStarted,
    UsageRecorded,
)
from agentkit.events.base import BaseEvent
from agentkit.loop.context import TurnContext
from agentkit.providers.base import ProviderEvent
from agentkit.tools.registry import ToolRegistry
from agentkit.tools.spec import RiskLevel

log = get_logger(__name__)


class StreamMux:
    def __init__(
        self,
        ctx: TurnContext,
        *,
        registry: ToolRegistry | None = None,
    ) -> None:
        self._ctx = ctx
        self._message_id: MessageId = new_id(MessageId)
        self._registry = registry

    @property
    def message_id(self) -> MessageId:
        return self._message_id

    @property
    def sequence(self) -> int:
        """The next sequence number that would be emitted (peek, no advance)."""
        return self._ctx.event_sequence

    async def translate(
        self,
        provider_events: AsyncIterator[ProviderEvent],
    ) -> AsyncIterator[BaseEvent]:
        """Yield user-facing events for each provider event."""
        pending_tool_starts: dict[str, dict[str, Any]] = {}

        async for ev in provider_events:
            match ev.type:
                case "message_start":
                    yield self._mk(MessageStarted, message_id=self._message_id)

                case "text_delta":
                    yield self._mk(
                        TextDelta,
                        message_id=self._message_id,
                        delta=ev.delta,
                        block_index=ev.block_index,
                    )

                case "thinking_delta":
                    yield self._mk(
                        ThinkingDelta,
                        message_id=self._message_id,
                        delta=ev.delta,
                    )

                case "tool_call_start":
                    pending_tool_starts[ev.call_id] = {"tool_name": ev.tool_name}
                    # Consumers still see nothing until the call completes, but
                    # the chunk must produce *an* event: the stall timeout is
                    # measuring this iterator, and a tool call that generates
                    # for longer than the timeout is otherwise indistinguishable
                    # from a dead provider. See ProviderActivity.
                    yield self._mk(ProviderActivity)

                case "tool_call_delta":
                    # consumers don't see argument deltas; complete event carries final args
                    yield self._mk(ProviderActivity)

                case "tool_call_complete":
                    risk_value = self._risk_for(ev.tool_name)
                    yield self._mk(
                        ToolCallStarted,
                        call_id=ev.call_id,
                        tool_name=ev.tool_name,
                        arguments=ev.arguments,
                        risk=risk_value,
                    )
                    pending_tool_starts.pop(ev.call_id, None)

                case "message_complete":
                    yield self._mk(
                        MessageCompleted,
                        message_id=self._message_id,
                        finish_reason=ev.finish_reason,
                    )

                case "usage":
                    # Capture into ctx.metadata for backward compatibility — any consumer
                    # reading the legacy capture directly continues to work.
                    self._ctx.metadata.setdefault("usages", []).append(ev.usage)
                    # Public per-call event for ledger consumers.
                    yield self._mk(
                        UsageRecorded,
                        message_id=self._message_id,
                        model=ev.model,
                        usage=ev.usage,
                        provider_name=ev.provider_name,
                    )

                case "error":
                    code = (
                        ErrorCode.RATE_LIMITED
                        if ev.code == "rate_limited"
                        else ErrorCode.PROVIDER_FAULT
                    )
                    yield self._mk(
                        Errored,
                        code=code,
                        message=ev.message,
                        recoverable=ev.recoverable,
                    )

        # Provider-agnostic residue check. A start with no complete means the
        # consumer got NO event for that call at all — starts and deltas are
        # parked here by design. This is the one seam that sees the symptom for
        # every provider, including ones whose parser has no end-of-stream
        # handling. Keep it to a length check and one call: it runs at the end
        # of every stream, and anything that raises here breaks all streaming.
        if pending_tool_starts:
            log.warning(
                "tool_call_start_without_complete",
                count=len(pending_tool_starts),
                tools=[meta["tool_name"] for meta in pending_tool_starts.values()],
                session_id=str(self._ctx.session_id),
                turn_id=str(self._ctx.turn_id),
            )

    def _risk_for(self, tool_name: str) -> str:
        """Look up the registered tool's risk level; fall back to READ for unknown.

        Unknown-tool fallback matches the runtime behavior in tool_phase, which
        treats a missing spec as auto-deny rather than letting it through; the
        ``risk`` value here is informational for the consumer's UI.
        """
        if self._registry is None:
            return RiskLevel.READ.value
        for spec in self._registry.list_specs():
            if spec.name == tool_name:
                return spec.risk.value
        return RiskLevel.READ.value

    def _mk(self, cls: type[BaseEvent], **payload: Any) -> BaseEvent:
        return cls(
            event_id=new_id(EventId),
            session_id=self._ctx.session_id,
            turn_id=self._ctx.turn_id,
            ts=self._ctx.clock.now(),
            sequence=self._ctx.next_sequence(),
            **payload,
        )
