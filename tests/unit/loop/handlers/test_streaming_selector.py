"""When deps['provider_selector'] is set, the streaming handler resolves
the provider via selector(ctx) per iteration instead of deps['provider']."""

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


def _deps_with(provider, selector=None) -> dict:
    return {
        "provider": provider,
        "provider_selector": selector,
        "message_builder": MessageBuilder(model="fake/test", max_tokens=1000),
        "registry": ToolRegistry(),
        "system_blocks": [],
        "success_claim": None,
    }


def _assistant_text(ctx: TurnContext) -> str:
    return " ".join(
        block.text
        for m in ctx.history
        if m.role == MessageRole.ASSISTANT
        for block in m.content
        if isinstance(block, TextBlock)
    )


@pytest.mark.asyncio
async def test_handler_uses_selector_when_present():
    """When deps['provider_selector'] is set, selector(ctx) is called and its
    return value is used, even if deps['provider'] is set to something else."""
    provider_a = FakeProvider().script(FakeProvider.text("from-a"))
    provider_b = FakeProvider().script(FakeProvider.text("from-b"))
    selector_calls = []

    def selector(ctx: TurnContext) -> FakeProvider:
        selector_calls.append(ctx)
        return provider_b

    ctx = TurnContext.empty()
    ctx.add_message(_user_msg("hi"))

    deps = _deps_with(provider_a, selector=selector)
    await handle_streaming(ctx, deps)

    assert len(selector_calls) == 1
    text = _assistant_text(ctx)
    assert "from-b" in text
    assert "from-a" not in text


@pytest.mark.asyncio
async def test_handler_falls_back_to_static_provider_when_selector_none():
    """When deps['provider_selector'] is None, the handler uses deps['provider']
    (backward-compatible behavior)."""
    provider_a = FakeProvider().script(FakeProvider.text("from-a"))

    ctx = TurnContext.empty()
    ctx.add_message(_user_msg("hi"))

    deps = _deps_with(provider_a, selector=None)
    await handle_streaming(ctx, deps)

    text = _assistant_text(ctx)
    assert "from-a" in text


@pytest.mark.asyncio
async def test_handler_falls_back_when_selector_key_missing():
    """When deps['provider_selector'] is absent entirely (legacy callers that
    built deps before this feature existed), the handler uses deps['provider']."""
    provider_a = FakeProvider().script(FakeProvider.text("from-a"))

    ctx = TurnContext.empty()
    ctx.add_message(_user_msg("hi"))

    deps = {
        "provider": provider_a,
        # no provider_selector key at all
        "message_builder": MessageBuilder(model="fake/test", max_tokens=1000),
        "registry": ToolRegistry(),
        "system_blocks": [],
        "success_claim": None,
    }
    await handle_streaming(ctx, deps)

    text = _assistant_text(ctx)
    assert "from-a" in text
