"""Targeted tests for stream parser edge cases not covered by request_builder tests."""

import importlib
import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import pytest

from agentkit.providers.base import (
    ErrorEvent,
    ToolCallComplete,
    ToolCallDelta,
    ToolCallStart,
    UsageEvent,
)
from agentkit.providers.openrouter.stream_parser import (
    INCOMPLETE_TOOL_CALL_CODE,
    INVALID_TOOL_ARGUMENTS_CODE,
    parse_openrouter_stream,
)
from agentkit.providers.openrouter.tool_name_codec import ToolNameCodec


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
    def __init__(self, choices: list[_Choice], usage: Any = None) -> None:
        self.choices = choices
        self.usage = usage


class _Usage:
    def __init__(self, prompt_tokens: int, completion_tokens: int, cached_tokens: int = 0) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.prompt_tokens_details = None


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


_PARSER_LOGGER = "agentkit.providers.openrouter.stream_parser"


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
    events = [ev async for ev in parse_openrouter_stream(_aiter(chunks), model="test/model")]
    types = [ev.type for ev in events]
    # We may see ToolCallStart and ToolCallDelta (legitimate), but NO ToolCallComplete.
    assert "tool_call_complete" not in types
    # MessageComplete still fires.
    assert types[-1] == "message_complete"


@pytest.mark.asyncio
@pytest.mark.parametrize("finish_reason", ["length", "stop", None])
async def test_dropped_pending_tool_emits_error_event_for_the_model(finish_reason):
    """Dropping the call is right; leaving the model to guess is not.

    Without an event the model sees its requested action simply vanish and goes
    on to narrate it as done. The ErrorEvent is the only thing that tells it the
    call had no effect and can be re-issued.
    """
    chunks: list[Any] = [
        _Chunk(
            [_Choice(_Delta(tool_calls=[_ToolCallStreamChunk(0, "call_1", "rm_file", '{"path":')]))]
        ),
        _Chunk([_Choice(_Delta(), finish_reason=finish_reason)]),
    ]
    events = [ev async for ev in parse_openrouter_stream(_aiter(chunks), model="test/model")]

    errors = [ev for ev in events if isinstance(ev, ErrorEvent)]
    assert len(errors) == 1
    assert errors[0].code == INCOMPLETE_TOOL_CALL_CODE
    assert errors[0].recoverable is True
    # Names the call so a consumer can pair it with the ToolCallStart it saw,
    # and says plainly that nothing ran.
    assert "rm_file" in errors[0].message
    assert "call_1" in errors[0].message
    assert "NOT executed" in errors[0].message
    # Still no dispatchable event, and the error precedes message_complete.
    assert "tool_call_complete" not in [ev.type for ev in events]
    assert [ev.type for ev in events].index("error") < [ev.type for ev in events].index(
        "message_complete"
    )


@pytest.mark.asyncio
async def test_every_dropped_slot_gets_its_own_error_event():
    """Two truncated calls must produce two errors — a single summary event
    would leave the model unsure which of its calls survived."""
    chunks: list[Any] = [
        _Chunk(
            [
                _Choice(
                    _Delta(
                        tool_calls=[
                            _ToolCallStreamChunk(0, "call_1", "rm_file", '{"path":'),
                            _ToolCallStreamChunk(1, "call_2", "send_mail", '{"to":'),
                        ]
                    )
                )
            ]
        ),
        _Chunk([_Choice(_Delta(), finish_reason="length")]),
    ]
    events = [ev async for ev in parse_openrouter_stream(_aiter(chunks), model="test/model")]
    errors = [ev for ev in events if isinstance(ev, ErrorEvent)]
    assert len(errors) == 2
    joined = " ".join(e.message for e in errors)
    assert "rm_file" in joined
    assert "send_mail" in joined


@pytest.mark.asyncio
@pytest.mark.parametrize("finish_reason", ["length", "stop", None])
async def test_dropped_pending_tools_emit_warning_log(caplog, finish_reason):
    """The drop is contract (see the test above); the SILENCE is the bug. A
    discarded pending tool call must leave exactly one greppable WARNING naming
    the finish_reason, the count, and whether args had actually arrived."""
    chunks: list[Any] = [
        _Chunk(
            [
                _Choice(
                    _Delta(tool_calls=[_ToolCallStreamChunk(0, "call_1", "rm_file", '{"path":1}')])
                )
            ]
        ),
        _Chunk([_Choice(_Delta(), finish_reason=finish_reason)]),
    ]
    with caplog.at_level(logging.WARNING, logger=_PARSER_LOGGER):
        events = [
            ev
            async for ev in parse_openrouter_stream(
                _aiter(chunks), model="test/model", session_id="sess-1"
            )
        ]

    # Contract re-pinned beside the log: still dropped, message still completes.
    assert "tool_call_complete" not in [ev.type for ev in events]

    records = [r for r in caplog.records if r.message == "openrouter.pending_tool_calls_dropped"]
    assert len(records) == 1, f"expected one drop warning, got {len(records)}"
    rec = records[0]
    assert rec.levelno == logging.WARNING
    assert rec.finish_reason == finish_reason
    assert rec.dropped_count == 1
    assert rec.model == "test/model"
    assert rec.session_id == "sess-1"
    assert rec.tools == [{"name": "rm_file", "has_id": True, "args_buf_len": len('{"path":1}')}]


@pytest.mark.asyncio
async def test_no_drop_warning_on_the_happy_path(caplog):
    """The drop warning must not fire when the flush actually happens — otherwise
    it is noise on every successful tool call and nobody will grep for it."""
    chunks: list[Any] = [
        _Chunk([_Choice(_Delta(tool_calls=[_ToolCallStreamChunk(0, "call_1", "add", "{}")]))]),
        _Chunk([_Choice(_Delta(), finish_reason="tool_calls")]),
    ]
    with caplog.at_level(logging.WARNING, logger=_PARSER_LOGGER):
        events = [ev async for ev in parse_openrouter_stream(_aiter(chunks), model="test/model")]
    assert [ev.type for ev in events].count("tool_call_complete") == 1
    assert [r.message for r in caplog.records if r.name == _PARSER_LOGGER] == []


@pytest.mark.asyncio
async def test_nameless_slot_with_args_is_skipped_and_logged(caplog):
    """Arguments arrived, the name delta never did. Nothing is dispatchable, so
    the slot is skipped — but no ToolCallStart was emitted for it either, and
    that asymmetry has to be visible somewhere."""
    chunks: list[Any] = [
        _Chunk([_Choice(_Delta(tool_calls=[_ToolCallStreamChunk(0, "call_1", None, '{"a":1}')]))]),
        _Chunk([_Choice(_Delta(), finish_reason="tool_calls")]),
    ]
    with caplog.at_level(logging.WARNING, logger=_PARSER_LOGGER):
        events = [ev async for ev in parse_openrouter_stream(_aiter(chunks), model="test/model")]

    types = [ev.type for ev in events]
    assert "tool_call_complete" not in types, "a nameless slot must never be dispatched"
    assert "tool_call_start" not in types, "Start only fires when a name arrives"

    records = [r for r in caplog.records if r.message == "openrouter.nameless_tool_slot_skipped"]
    assert len(records) == 1
    assert records[0].levelno == logging.WARNING
    assert records[0].args_buf_nonempty is True
    assert records[0].args_buf_len == len('{"a":1}')
    assert records[0].has_id is True

    # ...and the model is told, not just the log file.
    errors = [ev for ev in events if isinstance(ev, ErrorEvent)]
    assert len(errors) == 1
    assert errors[0].code == INCOMPLETE_TOOL_CALL_CODE
    assert errors[0].recoverable is True
    assert "NOT executed" in errors[0].message


@pytest.mark.asyncio
@pytest.mark.parametrize("nameless_first", [True, False])
async def test_named_and_nameless_slots_mixed_flush(caplog, nameless_first):
    """One bad slot must not take the batch down with it: the named sibling in
    the same message still flushes.

    Both orderings matter. The flush is a generator, so a bad slot in the SECOND
    position cannot retract a complete already yielded from the first — only the
    nameless-first ordering can catch an over-eager ``return`` (or a raise) that
    abandons the rest of the batch. A single-order version of this test passes
    against exactly that bug.
    """
    named = _ToolCallStreamChunk(0, "call_1", "add", '{"a":1}')
    nameless = _ToolCallStreamChunk(1, "call_2", None, '{"b":2}')
    if nameless_first:
        named.index, nameless.index = 1, 0
    ordered = [nameless, named] if nameless_first else [named, nameless]
    chunks: list[Any] = [
        _Chunk([_Choice(_Delta(tool_calls=ordered))]),
        _Chunk([_Choice(_Delta(), finish_reason="tool_calls")]),
    ]
    with caplog.at_level(logging.WARNING, logger=_PARSER_LOGGER):
        events = [ev async for ev in parse_openrouter_stream(_aiter(chunks), model="test/model")]

    completes = [ev for ev in events if isinstance(ev, ToolCallComplete)]
    assert [ev.tool_name for ev in completes] == ["add"]
    assert completes[0].arguments == {"a": 1}
    assert [r.message for r in caplog.records if r.name == _PARSER_LOGGER] == [
        "openrouter.nameless_tool_slot_skipped"
    ]


@pytest.mark.asyncio
async def test_unparseable_args_are_rejected_never_defaulted_to_empty(caplog):
    """Irreparable argument JSON must NOT be dispatched with ``arguments == {}``.

    ``{}`` is not a neutral value — for ``rm_file`` or "delete everything
    matching filter X" it is the widest possible reading of an intent we failed
    to decode, and the model never learns its JSON was broken. The call is
    refused and replaced by a retryable ErrorEvent naming the tool.
    """
    chunks: list[Any] = [
        _Chunk(
            [
                _Choice(
                    _Delta(
                        tool_calls=[
                            _ToolCallStreamChunk(0, "call_1", "rm_file", "not json at all !!!")
                        ]
                    )
                )
            ]
        ),
        _Chunk([_Choice(_Delta(), finish_reason="tool_calls")]),
    ]
    with caplog.at_level(logging.WARNING, logger=_PARSER_LOGGER):
        events = [ev async for ev in parse_openrouter_stream(_aiter(chunks), model="test/model")]

    assert [ev for ev in events if isinstance(ev, ToolCallComplete)] == []

    errors = [ev for ev in events if isinstance(ev, ErrorEvent)]
    assert len(errors) == 1
    assert errors[0].code == INVALID_TOOL_ARGUMENTS_CODE
    assert errors[0].recoverable is True
    assert "rm_file" in errors[0].message
    assert "call_1" in errors[0].message
    assert "not valid JSON" in errors[0].message

    records = [
        r for r in caplog.records if r.message == "openrouter.tool_args_unparseable_rejected"
    ]
    assert len(records) == 1
    assert records[0].levelno == logging.WARNING
    assert records[0].tool_name == "rm_file"
    assert records[0].args_buf_len == len("not json at all !!!")


@pytest.mark.asyncio
@pytest.mark.parametrize("args_json", ["[1, 2]", "null", '"just a string"', "42"])
async def test_non_object_arguments_are_rejected_too(args_json):
    """Valid JSON that is not an argument object is just as much a decode
    failure — it used to be silently coerced to ``{}`` by the isinstance check."""
    chunks: list[Any] = [
        _Chunk(
            [_Choice(_Delta(tool_calls=[_ToolCallStreamChunk(0, "call_1", "rm_file", args_json)]))]
        ),
        _Chunk([_Choice(_Delta(), finish_reason="tool_calls")]),
    ]
    events = [ev async for ev in parse_openrouter_stream(_aiter(chunks), model="test/model")]
    assert [ev for ev in events if isinstance(ev, ToolCallComplete)] == []
    errors = [ev for ev in events if isinstance(ev, ErrorEvent)]
    assert [e.code for e in errors] == [INVALID_TOOL_ARGUMENTS_CODE]


@pytest.mark.asyncio
async def test_no_arguments_at_all_is_a_legitimate_empty_call():
    """An empty argument buffer is a *parsed* empty object, not a guess: a tool
    with no parameters streams no argument deltas. Rejecting it would break
    every zero-arg tool, so this must stay dispatchable."""
    chunks: list[Any] = [
        _Chunk([_Choice(_Delta(tool_calls=[_ToolCallStreamChunk(0, "call_1", "ping", None)]))]),
        _Chunk([_Choice(_Delta(), finish_reason="tool_calls")]),
    ]
    events = [ev async for ev in parse_openrouter_stream(_aiter(chunks), model="test/model")]
    completes = [ev for ev in events if isinstance(ev, ToolCallComplete)]
    assert len(completes) == 1
    assert completes[0].arguments == {}
    assert [ev for ev in events if isinstance(ev, ErrorEvent)] == []


@pytest.mark.asyncio
async def test_id_less_tool_call_uses_one_fallback_call_id_across_all_events():
    """Start, delta and complete must agree on the synthesized id.

    They used to disagree (``call_0`` vs ``""``), so StreamMux's residue check
    saw a start that never completed and the consumer saw two unrelated calls.
    """
    chunks: list[Any] = [
        _Chunk([_Choice(_Delta(tool_calls=[_ToolCallStreamChunk(0, None, "add", '{"a":1}')]))]),
        _Chunk([_Choice(_Delta(), finish_reason="tool_calls")]),
    ]
    events = [ev async for ev in parse_openrouter_stream(_aiter(chunks), model="test/model")]
    ids = {
        ev.call_id
        for ev in events
        if isinstance(ev, ToolCallStart | ToolCallDelta | ToolCallComplete)
    }
    assert ids == {"call_0"}


@pytest.mark.asyncio
async def test_wire_tool_name_is_decoded_to_canonical_name():
    chunks: list[Any] = [
        _Chunk(
            [_Choice(_Delta(tool_calls=[_ToolCallStreamChunk(0, "call_1", "acme__search", "{}")]))]
        ),
        _Chunk([_Choice(_Delta(), finish_reason="tool_calls")]),
    ]
    codec = ToolNameCodec.from_names(["acme.search"])
    events = [
        ev
        async for ev in parse_openrouter_stream(
            _aiter(chunks), model="openai/gpt-5.6-luna", name_codec=codec
        )
    ]
    starts = [ev for ev in events if isinstance(ev, ToolCallStart)]
    completes = [ev for ev in events if isinstance(ev, ToolCallComplete)]
    assert [ev.tool_name for ev in starts + completes] == [
        "acme.search",
        "acme.search",
    ]


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
    events = [ev async for ev in parse_openrouter_stream(_aiter(chunks), model="test/model")]
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
    events = [ev async for ev in parse_openrouter_stream(_aiter(chunks), model="test/model")]
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
    events = [ev async for ev in parse_openrouter_stream(_aiter(chunks), model="openai/gpt-5")]
    thinking = [ev for ev in events if ev.type == "thinking_delta"]
    assert len(thinking) == 1
    assert thinking[0].delta == "hmm"


@pytest.mark.asyncio
async def test_parser_stamps_model_and_provider_name_on_usage_event():
    """The parser must thread its ``model`` kwarg + a fixed
    ``provider_name='openrouter'`` onto every UsageEvent it yields."""
    chunks: list[Any] = [
        _Chunk([_Choice(_Delta(content="hi"))]),
        _Chunk([_Choice(_Delta(), finish_reason="stop")]),
        # Usage arrives in a no-choices chunk (OpenAI stream_options pattern).
        _Chunk([], usage=_Usage(prompt_tokens=10, completion_tokens=5)),
    ]
    events = [ev async for ev in parse_openrouter_stream(_aiter(chunks), model="openai/gpt-5")]
    usage_events = [e for e in events if isinstance(e, UsageEvent)]
    assert len(usage_events) == 1
    assert usage_events[0].model == "openai/gpt-5"
    assert usage_events[0].provider_name == "openrouter"
    assert usage_events[0].usage.input_tokens == 10
    assert usage_events[0].usage.output_tokens == 5


@pytest.mark.asyncio
async def test_usage_captured_when_arrives_with_non_empty_choices():
    """OpenRouter delivers ``usage`` on the SAME chunk as the last delta +
    ``finish_reason`` (not on a follow-up no-choices chunk like the OpenAI
    canonical pattern). Live probes against deepseek-chat-v3.1,
    deepseek-v3.1-terminus, gemini-2.5-flash, and gemini-2.5-flash-lite-preview
    all emit this shape; the parser must capture usage on every chunk, not only
    when ``choices`` is empty. A downstream consumer's deploy once left ``usage_ledger``
    empty for 4+ hours because the original guard discarded these chunks.
    """
    chunks: list[Any] = [
        _Chunk([_Choice(_Delta(content="hi"))]),
        # Real OpenRouter final chunk: choices=non-empty + finish_reason + usage.
        _Chunk(
            [_Choice(_Delta(), finish_reason="stop")],
            usage=_Usage(prompt_tokens=12, completion_tokens=2),
        ),
    ]
    events = [
        ev
        async for ev in parse_openrouter_stream(_aiter(chunks), model="deepseek/deepseek-chat-v3.1")
    ]
    usage_events = [e for e in events if isinstance(e, UsageEvent)]
    assert len(usage_events) == 1, (
        f"Expected one UsageEvent for OpenRouter's final-chunk shape, got {len(usage_events)}. "
        "Usage on a chunk with non-empty choices was previously dropped."
    )
    assert usage_events[0].usage.input_tokens == 12
    assert usage_events[0].usage.output_tokens == 2
    assert usage_events[0].model == "deepseek/deepseek-chat-v3.1"


@pytest.mark.asyncio
async def test_translator_in_traced_when_session_allowlisted(monkeypatch, tmp_path):
    """Each TextDelta yielded by the parser must emit a ``translator_in`` JSONL
    line when ``session_id`` is allowlisted via STREAM_TRACE_SESSIONS. This is the
    upstream checkpoint that lets us localize a downstream consumer's chat truncation bugs."""
    sid = "drammen-trace-session"
    monkeypatch.setenv("STREAM_TRACE_SESSIONS", sid)
    monkeypatch.setenv("STREAM_TRACE_DIR", str(tmp_path))
    # Module-scoped imports reloaded so the env we just set takes effect.
    import agentkit._stream_trace as _trace_mod
    import agentkit.providers.openrouter.stream_parser as _parser_mod

    importlib.reload(_trace_mod)
    importlib.reload(_parser_mod)

    chunks: list[Any] = [
        _Chunk([_Choice(_Delta(content="Hei "))]),
        _Chunk([_Choice(_Delta(content="verden"))]),
        _Chunk([_Choice(_Delta(), finish_reason="stop")]),
    ]
    events = [
        ev
        async for ev in _parser_mod.parse_openrouter_stream(
            _aiter(chunks), model="test/m", session_id=sid
        )
    ]
    text = [ev for ev in events if ev.type == "text_delta"]
    assert [t.delta for t in text] == ["Hei ", "verden"]

    path = tmp_path / f"{sid}.jsonl"
    assert path.exists()
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    records = [json.loads(line) for line in lines]
    assert [r["checkpoint"] for r in records] == ["translator_in", "translator_in"]
    assert [r["content_repr"] for r in records] == [repr("Hei "), repr("verden")]
    assert all(r["extra"]["model"] == "test/m" for r in records)


@pytest.mark.asyncio
async def test_no_trace_when_session_unset(monkeypatch, tmp_path):
    """Parser must be a no-op (no file written) when session_id is None or unlisted."""
    monkeypatch.setenv("STREAM_TRACE_SESSIONS", "only-other-session")
    monkeypatch.setenv("STREAM_TRACE_DIR", str(tmp_path))
    # Module-scoped imports reloaded so the env we just set takes effect.
    import agentkit._stream_trace as _trace_mod
    import agentkit.providers.openrouter.stream_parser as _parser_mod

    importlib.reload(_trace_mod)
    importlib.reload(_parser_mod)

    chunks: list[Any] = [
        _Chunk([_Choice(_Delta(content="hi"))]),
        _Chunk([_Choice(_Delta(), finish_reason="stop")]),
    ]
    _ = [
        ev
        async for ev in _parser_mod.parse_openrouter_stream(
            _aiter(chunks), model="m", session_id=None
        )
    ]
    _ = [
        ev
        async for ev in _parser_mod.parse_openrouter_stream(
            _aiter(chunks), model="m", session_id="not-listed"
        )
    ]
    assert list(tmp_path.iterdir()) == []
