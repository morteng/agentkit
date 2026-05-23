"""Tests for Anthropic stream parser — event mapping and UsageEvent stamping."""

from collections.abc import AsyncIterator
from typing import Any

import pytest

from agentkit.providers.anthropic.stream_parser import parse_anthropic_stream
from agentkit.providers.base import UsageEvent

# ---------------------------------------------------------------------------
# Fake Anthropic SDK event objects
# ---------------------------------------------------------------------------


class _MessageUsage:
    def __init__(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_read_input_tokens: int = 0,
        cache_creation_input_tokens: int = 0,
    ) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_read_input_tokens = cache_read_input_tokens
        self.cache_creation_input_tokens = cache_creation_input_tokens


class _Message:
    def __init__(self, usage: _MessageUsage | None = None) -> None:
        self.usage = usage


class _MessageStartEvent:
    type = "message_start"

    def __init__(self, usage: _MessageUsage | None = None) -> None:
        self.message = _Message(usage=usage)


class _DeltaStopUsage:
    """Mimics the Anthropic SDK's MessageDeltaUsage — only output_tokens present."""

    def __init__(self, output_tokens: int = 10) -> None:
        self.output_tokens = output_tokens
        # The real SDK message_delta usage object does NOT expose input_tokens;
        # omitting it here ensures _parse_message_delta_usage falls back to
        # the prior value captured at message_start.


class _DeltaStop:
    def __init__(self, stop_reason: str = "end_turn") -> None:
        self.stop_reason = stop_reason


class _MessageDeltaEvent:
    type = "message_delta"

    def __init__(self, stop_reason: str = "end_turn", output_tokens: int = 10) -> None:
        self.delta = _DeltaStop(stop_reason=stop_reason)
        self.usage = _DeltaStopUsage(output_tokens=output_tokens)


class _MessageStopEvent:
    type = "message_stop"


async def _aiter(items: list[Any]) -> AsyncIterator[Any]:
    for it in items:
        yield it


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_anthropic_stamps_model_and_provider_name_on_usage_event():
    """The Anthropic stream parser must stamp the model identifier and
    provider_name='anthropic' onto every UsageEvent it yields, so
    cost-ledger consumers can attribute spend without inspecting the
    originating ProviderRequest."""
    events_in: list[Any] = [
        _MessageStartEvent(usage=_MessageUsage(input_tokens=20)),
        _MessageDeltaEvent(stop_reason="end_turn", output_tokens=15),
        _MessageStopEvent(),
    ]

    events_out = [
        ev
        async for ev in parse_anthropic_stream(_aiter(events_in), model="anthropic/claude-opus-4-7")
    ]

    usage_events = [e for e in events_out if isinstance(e, UsageEvent)]
    assert len(usage_events) == 1, "Expected exactly one UsageEvent"
    assert usage_events[0].model == "anthropic/claude-opus-4-7"
    assert usage_events[0].provider_name == "anthropic"
    assert usage_events[0].usage.input_tokens == 20
    assert usage_events[0].usage.output_tokens == 15
