"""StreamMux — translate ProviderEvents into user-facing Events.

Owns the per-turn ``sequence`` counter and the assistant message_id used by
streaming events. Deferred ``ToolCallStarted`` (with arguments) emits when the
provider sends ``tool_call_complete`` so the consumer sees a single event with
the full args, not a deluge of partial-JSON deltas.
"""

from collections.abc import AsyncIterator
from typing import Any

from agentkit._ids import EventId, MessageId, new_id
from agentkit.events import (
    ErrorCode,
    Errored,
    MessageCompleted,
    MessageStarted,
    TextDelta,
    ThinkingDelta,
    ToolCallStarted,
)
from agentkit.events.base import BaseEvent
from agentkit.loop.context import TurnContext
from agentkit.providers.base import ProviderEvent
from agentkit.tools.spec import RiskLevel


class StreamMux:
    def __init__(self, ctx: TurnContext, *, sequence_start: int) -> None:
        self._ctx = ctx
        self._seq = sequence_start
        self._message_id: MessageId = new_id(MessageId)

    @property
    def message_id(self) -> MessageId:
        return self._message_id

    @property
    def sequence(self) -> int:
        return self._seq

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

                case "tool_call_delta":
                    pass  # consumers don't see argument deltas; complete event carries final args

                case "tool_call_complete":
                    yield self._mk(
                        ToolCallStarted,
                        call_id=ev.call_id,
                        tool_name=ev.tool_name,
                        arguments=ev.arguments,
                        risk=RiskLevel.READ.value,
                    )
                    pending_tool_starts.pop(ev.call_id, None)

                case "message_complete":
                    yield self._mk(
                        MessageCompleted,
                        message_id=self._message_id,
                        finish_reason=ev.finish_reason,
                    )

                case "usage":
                    # Usage is captured into TurnMetrics later; not surfaced standalone.
                    self._ctx.metadata.setdefault("usages", []).append(ev.usage)

                case "error":
                    yield self._mk(
                        Errored,
                        code=ErrorCode.PROVIDER_FAULT,
                        message=ev.message,
                        recoverable=ev.recoverable,
                    )

    def _mk(self, cls: type[BaseEvent], **payload: Any) -> BaseEvent:
        seq = self._seq
        self._seq += 1
        return cls(
            event_id=new_id(EventId),
            session_id=self._ctx.session_id,
            turn_id=self._ctx.turn_id,
            ts=self._ctx.clock.now(),
            sequence=seq,
            **payload,
        )
