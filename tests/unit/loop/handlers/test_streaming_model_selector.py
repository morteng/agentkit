"""When deps['model_selector'] is set, the streaming handler resolves the
model via selector(ctx) per iteration and threads it into MessageBuilder
as a model_override, so ProviderRequest.model reflects the current-iteration
model rather than the session's constructor-time model.

Orthogonal to provider_selector: model_selector swaps the model_id;
provider_selector swaps the Provider instance (e.g. for per-tier
reasoning_effort). Pikkolo's tier router sets both off a shared tracker
so quality_model_id is actually called when the tier escalates.
"""

from datetime import UTC, datetime

import pytest

from agentkit._content import TextBlock
from agentkit._ids import MessageId, SessionId, new_id
from agentkit._messages import Message, MessageRole
from agentkit.loop.context import TurnContext
from agentkit.loop.handlers.streaming import handle_streaming
from agentkit.loop.message_builder import MessageBuilder
from agentkit.providers.fakes import FakeProvider
from agentkit.tools.registry import ToolRegistry


def _user_msg(text: str) -> Message:
    return Message(
        id=new_id(MessageId),
        session_id=new_id(SessionId),
        role=MessageRole.USER,
        content=[TextBlock(text=text)],
        created_at=datetime.now(UTC),
    )


def _deps_with(provider, *, model_selector=None) -> dict:
    return {
        "provider": provider,
        "provider_selector": None,
        "model_selector": model_selector,
        "message_builder": MessageBuilder(model="fake/standard", max_tokens=1000),
        "registry": ToolRegistry(),
        "system_blocks": [],
        "success_claim": None,
    }


@pytest.mark.asyncio
async def test_handler_uses_model_selector_when_present():
    """When deps['model_selector'] is set, selector(ctx) is called and its
    return value lands on the ProviderRequest.model — not the constructor-time
    model on the MessageBuilder.

    The FakeProvider stamps the request it received onto the asserter via
    closure capture so we can verify what model_id actually went to .stream()."""
    captured: list[str] = []

    class _RequestCapturingProvider(FakeProvider):
        async def stream(self, request):  # type: ignore[override]
            captured.append(request.model)
            async for event in super().stream(request):
                yield event

    provider = _RequestCapturingProvider().script(FakeProvider.text("ok"))

    def model_selector(ctx: TurnContext) -> str:
        return "fake/quality"

    ctx = TurnContext.empty()
    ctx.add_message(_user_msg("hi"))

    deps = _deps_with(provider, model_selector=model_selector)
    await handle_streaming(ctx, deps)

    assert captured == ["fake/quality"]


@pytest.mark.asyncio
async def test_handler_falls_back_to_builder_model_when_selector_none():
    """When deps['model_selector'] is None, request.model comes from the
    MessageBuilder's constructor-time model (backward-compatible)."""
    captured: list[str] = []

    class _RequestCapturingProvider(FakeProvider):
        async def stream(self, request):  # type: ignore[override]
            captured.append(request.model)
            async for event in super().stream(request):
                yield event

    provider = _RequestCapturingProvider().script(FakeProvider.text("ok"))

    ctx = TurnContext.empty()
    ctx.add_message(_user_msg("hi"))

    deps = _deps_with(provider, model_selector=None)
    await handle_streaming(ctx, deps)

    assert captured == ["fake/standard"]


@pytest.mark.asyncio
async def test_handler_falls_back_when_model_selector_key_missing():
    """Legacy callers that built deps before this feature existed don't have
    the 'model_selector' key at all — handler must still use the builder model."""
    captured: list[str] = []

    class _RequestCapturingProvider(FakeProvider):
        async def stream(self, request):  # type: ignore[override]
            captured.append(request.model)
            async for event in super().stream(request):
                yield event

    provider = _RequestCapturingProvider().script(FakeProvider.text("ok"))

    ctx = TurnContext.empty()
    ctx.add_message(_user_msg("hi"))

    deps = {
        "provider": provider,
        "provider_selector": None,
        # no model_selector key at all
        "message_builder": MessageBuilder(model="fake/standard", max_tokens=1000),
        "registry": ToolRegistry(),
        "system_blocks": [],
        "success_claim": None,
    }
    await handle_streaming(ctx, deps)

    assert captured == ["fake/standard"]


@pytest.mark.asyncio
async def test_model_and_provider_selectors_compose():
    """Both selectors fire per iteration: provider_selector picks the Provider,
    model_selector picks the model_id. They're independent inputs to the
    request — Pikkolo's tier router uses both off a shared tracker so the
    model on the wire matches the chosen tier."""
    captured: list[tuple[str, str]] = []

    class _RequestCapturingProvider(FakeProvider):
        def __init__(self, tag: str) -> None:
            super().__init__()
            self._tag = tag

        async def stream(self, request):  # type: ignore[override]
            captured.append((self._tag, request.model))
            async for event in super().stream(request):
                yield event

    provider_standard = _RequestCapturingProvider("standard").script(FakeProvider.text("ok"))
    provider_quality = _RequestCapturingProvider("quality").script(FakeProvider.text("ok"))

    def provider_selector(ctx: TurnContext):
        return provider_quality

    def model_selector(ctx: TurnContext) -> str:
        return "fake/quality"

    ctx = TurnContext.empty()
    ctx.add_message(_user_msg("hi"))

    deps = {
        "provider": provider_standard,
        "provider_selector": provider_selector,
        "model_selector": model_selector,
        "message_builder": MessageBuilder(model="fake/standard", max_tokens=1000),
        "registry": ToolRegistry(),
        "system_blocks": [],
        "success_claim": None,
    }
    await handle_streaming(ctx, deps)

    # Quality provider was picked AND quality model was stamped — neither
    # the static provider nor the builder model leaked through.
    assert captured == [("quality", "fake/quality")]
