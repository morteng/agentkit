"""Tests for Anthropic stream parser — event mapping and UsageEvent stamping."""

import logging
from collections.abc import AsyncIterator
from typing import Any

import pytest

from agentkit.providers.anthropic.stream_parser import parse_anthropic_stream
from agentkit.providers.base import ErrorEvent, ToolCallComplete, UsageEvent
from agentkit.providers.tool_call_errors import INVALID_TOOL_ARGUMENTS_CODE

_PARSER_LOGGER = "agentkit.providers.anthropic.stream_parser"

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


class _ToolUseBlock:
    type = "tool_use"

    def __init__(self, block_id: str, name: str) -> None:
        self.id = block_id
        self.name = name


class _ContentBlockStartEvent:
    type = "content_block_start"

    def __init__(self, index: int, block_id: str, name: str) -> None:
        self.index = index
        self.content_block = _ToolUseBlock(block_id, name)


class _InputJSONDelta:
    type = "input_json_delta"

    def __init__(self, partial_json: str) -> None:
        self.partial_json = partial_json


class _ContentBlockDeltaEvent:
    type = "content_block_delta"

    def __init__(self, index: int, partial_json: str) -> None:
        self.index = index
        self.delta = _InputJSONDelta(partial_json)


class _ContentBlockStopEvent:
    type = "content_block_stop"

    def __init__(self, index: int) -> None:
        self.index = index


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


@pytest.mark.asyncio
async def test_truncated_tool_json_is_rejected_never_defaulted_to_empty(caplog):
    """Truncated tool JSON must not be dispatched with ``arguments == {}``.

    This test used to pin the opposite: the parser emitted the call anyway on a
    corrupted intent, and the assertion existed as a tripwire for whenever the
    two providers were aligned. This is that alignment. ``rm_file`` with a
    truncated ``{"path": "/etc/`` is the whole argument — an empty argument set
    is not a neutral fallback, it is a different and more dangerous call than
    the one the model tried to make.

    Both parsers now resolve it the same way (see
    ``agentkit.providers.tool_call_errors``): no ``ToolCallComplete``, one
    recoverable ``ErrorEvent`` whose message is written for the model to read
    and re-issue from.
    """
    events_in: list[Any] = [
        _MessageStartEvent(usage=_MessageUsage(input_tokens=5)),
        _ContentBlockStartEvent(0, "toolu_1", "rm_file"),
        _ContentBlockDeltaEvent(0, '{"path": "/etc/'),  # stream truncated mid-JSON
        _ContentBlockStopEvent(0),
        _MessageDeltaEvent(stop_reason="max_tokens", output_tokens=3),
        _MessageStopEvent(),
    ]
    with caplog.at_level(logging.WARNING, logger=_PARSER_LOGGER):
        events_out = [
            ev async for ev in parse_anthropic_stream(_aiter(events_in), model="anthropic/claude")
        ]

    assert [e for e in events_out if isinstance(e, ToolCallComplete)] == []

    errors = [e for e in events_out if isinstance(e, ErrorEvent)]
    assert len(errors) == 1
    assert errors[0].code == INVALID_TOOL_ARGUMENTS_CODE
    assert errors[0].recoverable is True
    # Model-facing: it must name the call and say it did not run.
    assert "rm_file" in errors[0].message
    assert "NOT executed" in errors[0].message

    records = [r for r in caplog.records if r.message == "anthropic.tool_args_unparseable_rejected"]
    assert len(records) == 1
    assert records[0].levelno == logging.WARNING
    assert records[0].tool_name == "rm_file"
    assert records[0].args_buf_len == len('{"path": "/etc/')


@pytest.mark.asyncio
async def test_tool_json_that_is_not_an_object_is_rejected():
    """``json.loads`` accepts ``[1,2]`` / ``null`` / ``4``; a handler cannot.

    The old ``args: dict = json.loads(...)`` annotation was a lie for these —
    they parsed cleanly and were handed to the dispatcher as "arguments".
    """
    for payload in ("[1, 2]", "null", '"a string"', "42"):
        events_in: list[Any] = [
            _MessageStartEvent(usage=_MessageUsage(input_tokens=5)),
            _ContentBlockStartEvent(0, "toolu_1", "add"),
            _ContentBlockDeltaEvent(0, payload),
            _ContentBlockStopEvent(0),
            _MessageDeltaEvent(stop_reason="tool_use", output_tokens=3),
            _MessageStopEvent(),
        ]
        events_out = [
            ev async for ev in parse_anthropic_stream(_aiter(events_in), model="anthropic/claude")
        ]
        assert [e for e in events_out if isinstance(e, ToolCallComplete)] == [], payload
        errors = [e for e in events_out if isinstance(e, ErrorEvent)]
        assert len(errors) == 1, payload
        assert errors[0].code == INVALID_TOOL_ARGUMENTS_CODE


@pytest.mark.asyncio
async def test_well_formed_tool_json_does_not_log(caplog):
    """The coercion warning must stay quiet on the happy path."""
    events_in: list[Any] = [
        _MessageStartEvent(usage=_MessageUsage(input_tokens=5)),
        _ContentBlockStartEvent(0, "toolu_1", "add"),
        _ContentBlockDeltaEvent(0, '{"a": 1}'),
        _ContentBlockStopEvent(0),
        _MessageDeltaEvent(stop_reason="tool_use", output_tokens=3),
        _MessageStopEvent(),
    ]
    with caplog.at_level(logging.WARNING, logger=_PARSER_LOGGER):
        events_out = [
            ev async for ev in parse_anthropic_stream(_aiter(events_in), model="anthropic/claude")
        ]

    completes = [e for e in events_out if isinstance(e, ToolCallComplete)]
    assert completes[0].arguments == {"a": 1}
    assert [r.message for r in caplog.records if r.name == _PARSER_LOGGER] == []
