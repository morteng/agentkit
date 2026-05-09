"""Stream parser must use parse_tool_args_with_repair so malformed DeepSeek
tool-call args don't crash the stream."""

import json
from collections.abc import AsyncIterator
from typing import Any

import pytest

from agentkit.providers.openrouter.stream_parser import (
    parse_openrouter_stream,
    parse_tool_call_arguments,
)


def test_stream_parser_uses_repair_for_malformed_args():
    bad = '{"facts": Drammens Teater ble bygd i 1869, "category": "history"}'
    parsed = parse_tool_call_arguments(bad)
    assert parsed is not None
    assert "facts" in parsed


def test_stream_parser_returns_dict_for_well_formed_args():
    parsed = parse_tool_call_arguments('{"a": 1}')
    assert parsed == {"a": 1}


def test_stream_parser_empty_string_returns_empty_dict():
    parsed = parse_tool_call_arguments("")
    assert parsed == {}


def test_stream_parser_raises_on_unrecoverable_garbage():
    with pytest.raises(json.JSONDecodeError):
        parse_tool_call_arguments("not json at all !!!")


class _Delta:
    def __init__(
        self,
        content: str | None = None,
        tool_calls: list[Any] | None = None,
    ) -> None:
        self.content = content
        self.tool_calls = tool_calls
        self.reasoning_content = None
        self.reasoning = None


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
async def test_malformed_tool_args_are_repaired_through_stream():
    """End-to-end: malformed DeepSeek tool-call JSON is repaired at stream flush,
    producing a valid ToolCallComplete with recovered arguments instead of {}."""
    malformed = '{"facts": Drammens Teater ble bygd i 1869, "category": "history"}'
    chunks: list[Any] = [
        _Chunk(
            [
                _Choice(
                    _Delta(tool_calls=[_ToolCallStreamChunk(0, "call_1", "store_fact", malformed)])
                )
            ]
        ),
        _Chunk([_Choice(_Delta(), finish_reason="tool_calls")]),
    ]
    events = [ev async for ev in parse_openrouter_stream(_aiter(chunks))]
    tcc = [ev for ev in events if ev.type == "tool_call_complete"]
    assert len(tcc) == 1
    assert tcc[0].tool_name == "store_fact"
    # Repair should have recovered at minimum the "category" key with its proper string value.
    assert "category" in tcc[0].arguments
    assert tcc[0].arguments["category"] == "history"
