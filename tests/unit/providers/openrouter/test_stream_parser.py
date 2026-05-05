"""Targeted tests for stream parser edge cases not covered by request_builder tests."""

from collections.abc import AsyncIterator
from typing import Any

import pytest

from agentkit.providers.openrouter.stream_parser import parse_openrouter_stream


class _Delta:
    def __init__(
        self,
        content: str | None = None,
        tool_calls: list[Any] | None = None,
        reasoning_content: str | None = None,
        reasoning: str | None = None,
    ) -> None:
        self.content = content
        self.tool_calls = tool_calls
        self.reasoning_content = reasoning_content
        self.reasoning = reasoning


class _Choice:
    def __init__(self, delta: _Delta, finish_reason: str | None = None) -> None:
        self.delta = delta
        self.finish_reason = finish_reason


class _Chunk:
    def __init__(self, choices: list[_Choice]) -> None:
        self.choices = choices


class _ToolCallStreamChunk:
    def __init__(self, index: int, id: str | None, name: str | None, arguments: str | None) -> None:
        self.index = index
        self.id = id

        class _Fn:
            def __init__(self, name: str | None, arguments: str | None) -> None:
                self.name = name
                self.arguments = arguments

        self.function = _Fn(name, arguments)


async def _aiter(items: list[Any]) -> AsyncIterator[Any]:
    for it in items:
        yield it


@pytest.mark.asyncio
async def test_pending_tool_calls_are_dropped_when_stream_ends_abnormally():
    """If the stream terminates without finish_reason="tool_calls", any pending
    tool-call accumulation should NOT be emitted — partial args could be
    coerced to {} and lead to destructive tool execution."""
    chunks: list[Any] = [
        _Chunk(
            [_Choice(_Delta(tool_calls=[_ToolCallStreamChunk(0, "call_1", "rm_file", '{"path":')]))]
        ),
        # Stream ends with finish_reason="length" (truncation), NOT "tool_calls".
        _Chunk([_Choice(_Delta(), finish_reason="length")]),
    ]
    events = [ev async for ev in parse_openrouter_stream(_aiter(chunks))]
    types = [ev.type for ev in events]
    # We may see ToolCallStart and ToolCallDelta (legitimate), but NO ToolCallComplete.
    assert "tool_call_complete" not in types
    # MessageComplete still fires.
    assert types[-1] == "message_complete"


@pytest.mark.asyncio
async def test_pending_tool_calls_flush_when_finish_reason_is_tool_calls():
    """Happy path — finish_reason="tool_calls" means args are complete; emit ToolCallComplete."""
    chunks: list[Any] = [
        _Chunk(
            [
                _Choice(
                    _Delta(tool_calls=[_ToolCallStreamChunk(0, "call_1", "add", '{"a":1,"b":2}')])
                )
            ]
        ),
        _Chunk([_Choice(_Delta(), finish_reason="tool_calls")]),
    ]
    events = [ev async for ev in parse_openrouter_stream(_aiter(chunks))]
    tcc = [ev for ev in events if ev.type == "tool_call_complete"]
    assert len(tcc) == 1
    assert tcc[0].tool_name == "add"
    assert tcc[0].arguments == {"a": 1, "b": 2}


@pytest.mark.asyncio
async def test_reasoning_content_emitted_as_thinking_delta():
    """F1: OpenRouter reasoning_content (DeepSeek chain-of-thought) maps to ThinkingDelta."""
    chunks: list[Any] = [
        _Chunk([_Choice(_Delta(reasoning_content="thinking about it..."))]),
        _Chunk([_Choice(_Delta(reasoning_content=" still thinking"))]),
        _Chunk([_Choice(_Delta(content="Hi!"))]),
        _Chunk([_Choice(_Delta(), finish_reason="stop")]),
    ]
    events = [ev async for ev in parse_openrouter_stream(_aiter(chunks))]
    thinking = [ev for ev in events if ev.type == "thinking_delta"]
    text = [ev for ev in events if ev.type == "text_delta"]
    assert [t.delta for t in thinking] == ["thinking about it...", " still thinking"]
    assert [t.delta for t in text] == ["Hi!"]


@pytest.mark.asyncio
async def test_reasoning_field_alias_also_emits_thinking_delta():
    """F1: some OpenRouter providers use ``reasoning`` instead of ``reasoning_content``."""
    chunks: list[Any] = [
        _Chunk([_Choice(_Delta(reasoning="hmm"))]),
        _Chunk([_Choice(_Delta(content="ok"))]),
        _Chunk([_Choice(_Delta(), finish_reason="stop")]),
    ]
    events = [ev async for ev in parse_openrouter_stream(_aiter(chunks))]
    thinking = [ev for ev in events if ev.type == "thinking_delta"]
    assert len(thinking) == 1
    assert thinking[0].delta == "hmm"
