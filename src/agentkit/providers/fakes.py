"""FakeProvider — scriptable stream for tests.

Used everywhere in unit/e2e tests. Treats responses as a queue: each call to
``stream()`` consumes the next ``ScriptedResponse`` and yields its events.
"""

from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from ulid import ULID

from agentkit._content import TextBlock
from agentkit._messages import Message, Usage
from agentkit.providers.base import (
    ErrorEvent,
    MessageComplete,
    MessageStart,
    Provider,
    ProviderCapabilities,
    ProviderEvent,
    ProviderRequest,
    TextDelta,
    ToolCallComplete,
    ToolCallStart,
    UsageEvent,
)


@dataclass
class ScriptedResponse:
    """One full response from the fake. Either text or a single tool call."""

    kind: str  # "text" | "tool_call" | "error"
    text: str | None = None
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    error_code: str | None = None
    error_message: str | None = None
    usage: Usage | None = None


class FakeProvider(Provider):
    name = "fake"
    capabilities = ProviderCapabilities(
        supports_tool_use=True,
        supports_parallel_tools=True,
        supports_prompt_caching=False,
        supports_vision=True,
        supports_thinking=False,
        max_context_tokens=200_000,
        max_output_tokens=8_192,
    )

    def __init__(self) -> None:
        self._queue: deque[ScriptedResponse] = deque()

    # ---- Scripting helpers (declarative test fixture API) -------------------

    def script(self, *responses: ScriptedResponse) -> "FakeProvider":
        for r in responses:
            self._queue.append(r)
        return self

    @staticmethod
    def text(content: str, usage: Usage | None = None) -> ScriptedResponse:
        return ScriptedResponse(kind="text", text=content, usage=usage)

    @staticmethod
    def tool_call(
        name: str,
        args: dict[str, Any],
        usage: Usage | None = None,
    ) -> ScriptedResponse:
        return ScriptedResponse(kind="tool_call", tool_name=name, tool_args=args, usage=usage)

    @staticmethod
    def error(code: str, message: str) -> ScriptedResponse:
        return ScriptedResponse(kind="error", error_code=code, error_message=message)

    # ---- Provider protocol --------------------------------------------------

    async def stream(self, request: ProviderRequest) -> AsyncIterator[ProviderEvent]:
        if not self._queue:
            raise RuntimeError(
                "FakeProvider out of scripted responses. Add more via FakeProvider.script(...)."
            )
        response = self._queue.popleft()

        if response.kind == "error":
            yield ErrorEvent(
                code=response.error_code or "unknown",
                message=response.error_message or "",
                recoverable=False,
            )
            return

        yield MessageStart()

        if response.kind == "text":
            assert response.text is not None
            # Stream chunks of ~3 chars to mimic a real provider.
            for i in range(0, len(response.text), 3):
                yield TextDelta(delta=response.text[i : i + 3], block_index=0)
            yield UsageEvent(usage=response.usage or Usage(input_tokens=10, output_tokens=10))
            yield MessageComplete(finish_reason="end_turn")

        elif response.kind == "tool_call":
            assert response.tool_name is not None and response.tool_args is not None
            call_id = f"call_{ULID()}"
            yield ToolCallStart(call_id=call_id, tool_name=response.tool_name)
            yield ToolCallComplete(
                call_id=call_id,
                tool_name=response.tool_name,
                arguments=response.tool_args,
            )
            yield UsageEvent(usage=response.usage or Usage(input_tokens=10, output_tokens=5))
            yield MessageComplete(finish_reason="tool_use")

    def estimate_tokens(self, messages: list[Message]) -> int:
        return sum(
            len(b.text) // 4 for m in messages for b in m.content if isinstance(b, TextBlock)
        )

    def estimate_cost(self, usage: Usage) -> Decimal:
        return Decimal("0")
