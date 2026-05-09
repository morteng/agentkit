"""Fixtures for real-SDK contract tests.

These tests construct the actual provider SDK client (openai.AsyncOpenAI,
anthropic.AsyncAnthropic) with a stubbed httpx.AsyncClient transport, so
the SDK validates kwargs and serialization against its real schema without
requiring an API key or network access.
"""

import json
from collections.abc import Iterator
from typing import Any

import httpx
import pytest


def _empty_chat_completion_response() -> dict[str, Any]:
    return {
        "id": "chatcmpl-stub",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "stub",
        "choices": [
            {"index": 0, "delta": {}, "finish_reason": "stop"},
        ],
    }


def _stub_chat_handler(_request: httpx.Request) -> httpx.Response:
    """Return an empty SSE stream so SDK clients exit cleanly."""
    body = f"data: {json.dumps(_empty_chat_completion_response())}\n\ndata: [DONE]\n\n"
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        content=body.encode(),
    )


def _stub_anthropic_messages_handler(_request: httpx.Request) -> httpx.Response:
    """Return an empty Anthropic SSE stream so SDK clients exit cleanly."""
    body = (
        "event: message_start\n"
        'data: {"type":"message_start","message":{"id":"msg_stub","type":"message",'
        '"role":"assistant","content":[],"model":"stub","stop_reason":null,'
        '"stop_sequence":null,"usage":{"input_tokens":0,"output_tokens":0}}}\n\n'
        "event: message_stop\n"
        'data: {"type":"message_stop"}\n\n'
    )
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        content=body.encode(),
    )


@pytest.fixture
def stub_openai_transport() -> Iterator[httpx.MockTransport]:
    """httpx transport that answers /chat/completions with an empty stream."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/chat/completions"):
            return _stub_chat_handler(request)
        return httpx.Response(404)

    yield httpx.MockTransport(handler)


@pytest.fixture
def stub_anthropic_transport() -> Iterator[httpx.MockTransport]:
    """httpx transport that answers /v1/messages with an empty stream."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/v1/messages"):
            return _stub_anthropic_messages_handler(request)
        return httpx.Response(404)

    yield httpx.MockTransport(handler)


@pytest.fixture
def real_openai_client(stub_openai_transport: httpx.MockTransport):
    """Real openai.AsyncOpenAI with stubbed transport — validates kwargs against
    the SDK's real schema without an API key."""
    from openai import AsyncOpenAI

    return AsyncOpenAI(
        api_key="stub",
        base_url="http://stub.invalid/v1",
        http_client=httpx.AsyncClient(transport=stub_openai_transport),
    )


@pytest.fixture
def real_anthropic_client(stub_anthropic_transport: httpx.MockTransport):
    """Real anthropic.AsyncAnthropic with stubbed transport."""
    from anthropic import AsyncAnthropic

    return AsyncAnthropic(
        api_key="stub",
        base_url="http://stub.invalid",
        http_client=httpx.AsyncClient(transport=stub_anthropic_transport),
    )
