"""Forcing the finalize tool on a missing-finalize re-prompt.

When a turn ends without calling finalize (the model answered inline and went
quiet), agentkit re-prompts it to finalize. With
``force_finalize_on_missing_reprompt`` enabled, that re-prompt turn is
constrained to the finalize tool via ``tool_choice`` — so the model emits the
envelope immediately instead of spending another full free-form turn
(thinking / re-narrating) that would hold the consumer in a streaming state
for minutes (the v0.155.3 stuck-stream symptom).
"""

import asyncio
from datetime import UTC, datetime
from typing import Any

import pytest

from agentkit._content import TextBlock
from agentkit._ids import MessageId, SessionId, new_id
from agentkit._messages import Message, MessageRole
from agentkit.guards.finalize import StructuralFinalizeValidator
from agentkit.loop.context import TurnContext
from agentkit.loop.handlers.finalize_check import handle_finalize_check
from agentkit.loop.handlers.streaming import handle_streaming
from agentkit.loop.message_builder import MessageBuilder
from agentkit.providers.base import NamedToolChoice, ProviderRequest
from agentkit.providers.fakes import FakeProvider
from agentkit.tools.builtin.finalize import FINALIZE_SPEC, finalize_handler
from agentkit.tools.registry import ToolRegistry


def _user(text: str) -> Message:
    return Message(
        id=new_id(MessageId),
        session_id=new_id(SessionId),
        role=MessageRole.USER,
        content=[TextBlock(text=text)],
        created_at=datetime.now(UTC),
    )


# ---- finalize_check arms the force flag on the missing-finalize re-prompt ----


@pytest.mark.asyncio
async def test_missing_finalize_reprompt_arms_force_tool_choice_when_enabled():
    ctx = TurnContext.empty()
    ctx.add_message(_user("how many days to clear the backlog?"))
    deps = {
        "finalize_validator": StructuralFinalizeValidator(),
        "force_finalize_on_missing_reprompt": True,
    }
    await handle_finalize_check(ctx, deps)
    assert ctx.metadata.get("force_finalize_tool_choice") is True


@pytest.mark.asyncio
async def test_missing_finalize_reprompt_does_not_arm_force_by_default():
    ctx = TurnContext.empty()
    ctx.add_message(_user("how many days to clear the backlog?"))
    deps = {"finalize_validator": StructuralFinalizeValidator()}
    await handle_finalize_check(ctx, deps)
    assert "force_finalize_tool_choice" not in ctx.metadata


@pytest.mark.asyncio
async def test_rejected_finalize_retry_arms_force_tool_choice_when_enabled():
    """When a finalize envelope is rejected, the retry turn is also constrained."""
    ctx = TurnContext.empty()
    ctx.add_message(_user("do something"))
    ctx.finalize_called = True
    ctx.finalize_args = {
        "status": "done",
        "intent_kind": "action",
        "actions_performed": [],  # Rule 4 violation: empty_on_done
    }
    deps = {
        "finalize_validator": StructuralFinalizeValidator(),
        "force_finalize_on_missing_reprompt": True,
        "max_finalize_retries": 2,
    }
    await handle_finalize_check(ctx, deps)
    assert ctx.metadata.get("force_finalize_tool_choice") is True
    assert ctx.metadata.get("finalize_retries") == 1


@pytest.mark.asyncio
async def test_rejected_finalize_retry_does_not_arm_force_by_default():
    ctx = TurnContext.empty()
    ctx.add_message(_user("do something"))
    ctx.finalize_called = True
    ctx.finalize_args = {
        "status": "done",
        "intent_kind": "action",
        "actions_performed": [],
    }
    deps = {
        "finalize_validator": StructuralFinalizeValidator(),
        "max_finalize_retries": 2,
    }
    await handle_finalize_check(ctx, deps)
    assert "force_finalize_tool_choice" not in ctx.metadata
    assert ctx.metadata.get("finalize_retries") == 1


# ---- streaming applies the forced tool choice, one-shot ----


class _CapturingProvider(FakeProvider):
    """FakeProvider that records the last request it was asked to stream."""

    def __init__(self) -> None:
        super().__init__()
        self.last_request: ProviderRequest | None = None

    async def stream(self, request: ProviderRequest):  # type: ignore[override]
        self.last_request = request
        async for ev in super().stream(request):
            yield ev


def _finalize_registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register_builtin(FINALIZE_SPEC, finalize_handler)  # bare name "finalize"
    return reg


def _streaming_deps(provider: _CapturingProvider) -> dict[str, Any]:
    return {
        "provider": provider,
        "message_builder": MessageBuilder(model="m", max_tokens=128),
        "registry": _finalize_registry(),
        "system_blocks": [],
        "success_claim": None,
    }


@pytest.mark.asyncio
async def test_streaming_forces_finalize_tool_choice_when_flag_set():
    provider = _CapturingProvider()
    provider.script(FakeProvider.text("ok"))
    ctx = TurnContext.empty()
    ctx.add_message(_user("finalize now"))
    ctx.event_queue = asyncio.Queue()
    ctx.metadata["force_finalize_tool_choice"] = True

    await handle_streaming(ctx, _streaming_deps(provider))

    assert provider.last_request is not None
    assert isinstance(provider.last_request.tool_choice, NamedToolChoice)
    assert provider.last_request.tool_choice.name == "kit.finalize"
    # One-shot: the flag is consumed so the next iteration is unconstrained.
    assert "force_finalize_tool_choice" not in ctx.metadata


@pytest.mark.asyncio
async def test_streaming_leaves_tool_choice_auto_without_flag():
    provider = _CapturingProvider()
    provider.script(FakeProvider.text("ok"))
    ctx = TurnContext.empty()
    ctx.add_message(_user("hi"))
    ctx.event_queue = asyncio.Queue()

    await handle_streaming(ctx, _streaming_deps(provider))

    assert provider.last_request is not None
    assert provider.last_request.tool_choice == "auto"


@pytest.mark.asyncio
async def test_streaming_force_flag_without_finalize_tool_falls_back_to_auto():
    """No finalize tool registered -> can't force; fall back to an unconstrained turn."""
    provider = _CapturingProvider()
    provider.script(FakeProvider.text("ok"))
    ctx = TurnContext.empty()
    ctx.add_message(_user("finalize now"))
    ctx.event_queue = asyncio.Queue()
    ctx.metadata["force_finalize_tool_choice"] = True
    deps = _streaming_deps(provider)
    deps["registry"] = ToolRegistry()  # no finalize tool

    await handle_streaming(ctx, deps)

    assert provider.last_request is not None
    assert provider.last_request.tool_choice == "auto"
    assert "force_finalize_tool_choice" not in ctx.metadata


@pytest.mark.asyncio
async def test_the_forced_tool_must_be_one_the_request_actually_offers():
    """A `tool_selector` that drops the finalize tool must not leave the
    request naming it.

    The name used to be resolved from the REGISTRY while the request was built
    from whatever the selector left, so the two could disagree. A payload whose
    `tool_choice` names a tool absent from its own `tools` list is rejected or
    ignored by providers — and the flag has already been popped, so nothing
    tries again. The re-prompt budget is then spent on an unconstrained turn
    that is indistinguishable, from the outside, from a forced turn the model
    ignored.

    Asserted on what the provider RECEIVED, because the flag and the registry
    both look correct in the failing case; only the request disagrees.
    """
    provider = _CapturingProvider()
    provider.script(FakeProvider.text("ok"))
    ctx = TurnContext.empty()
    ctx.add_message(_user("finalize now"))
    ctx.event_queue = asyncio.Queue()
    ctx.metadata["force_finalize_tool_choice"] = True
    deps = _streaming_deps(provider)
    # The registry still HAS the finalize tool. The selector does not offer it.
    deps["tool_selector"] = lambda _ctx, _specs: []

    await handle_streaming(ctx, deps)

    assert provider.last_request is not None
    assert provider.last_request.tool_choice == "auto", (
        "tool_choice named a tool the request does not carry"
    )
    assert provider.last_request.tools == []
    assert "force_finalize_tool_choice" not in ctx.metadata


@pytest.mark.asyncio
async def test_a_selector_that_keeps_finalize_still_forces_it():
    """Control for the test above.

    Without this, "tool_choice is auto whenever a selector is installed" would
    satisfy it, and the fix would be indistinguishable from switching the
    forcing off for every consumer that uses a selector at all.
    """
    provider = _CapturingProvider()
    provider.script(FakeProvider.text("ok"))
    ctx = TurnContext.empty()
    ctx.add_message(_user("finalize now"))
    ctx.event_queue = asyncio.Queue()
    ctx.metadata["force_finalize_tool_choice"] = True
    deps = _streaming_deps(provider)
    deps["tool_selector"] = lambda _ctx, specs: list(specs)

    await handle_streaming(ctx, deps)

    assert provider.last_request is not None
    assert isinstance(provider.last_request.tool_choice, NamedToolChoice)
    assert provider.last_request.tool_choice.name == "kit.finalize"
    assert [t.name for t in provider.last_request.tools] == ["kit.finalize"]
