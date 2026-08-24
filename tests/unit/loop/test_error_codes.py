"""An exception chooses the ``ErrorCode`` the consumer sees.

``Orchestrator._record_turn_error`` is the single funnel every turn-ending
failure passes through, and it hard-coded ``ErrorCode.INTERNAL``. ``code`` is
the only structured field on ``Errored`` and the only one a consumer can safely
render — ``message`` is free text that may carry a provider's HTTP body — so
hard-coding it made a deliberate policy refusal indistinguishable from an
unhandled crash, and left pattern-matching exception names out of ``message``
as the only way to tell them apart.

These tests pin the resolution rule, the paths that must stay ``INTERNAL``, and
the seam that actually reported the bug: a firewall refusal raised inside a
provider, surfacing on the wire as ``POLICY_REFUSED``.
"""

import asyncio
from datetime import UTC, datetime

import pytest

from agentkit._content import TextBlock
from agentkit._ids import EventId, MessageId, SessionId, new_id
from agentkit._messages import Message, MessageRole
from agentkit.errors import (
    AgentkitError,
    ApprovalTimeout,
    CheckpointMissing,
    ConfigurationError,
    InvalidPhaseTransition,
    ProviderError,
    StoreError,
    ToolError,
    error_code_for,
)
from agentkit.events import EVENT_ADAPTER, ErrorCode, Errored
from agentkit.loop.context import TurnContext
from agentkit.loop.handlers.streaming import handle_streaming
from agentkit.loop.message_builder import MessageBuilder
from agentkit.loop.orchestrator import Loop
from agentkit.loop.phase import Phase
from agentkit.pii.firewall import Firewall
from agentkit.pii.policy import BlockedModelError, PiiPolicy, ZdrRouteUnavailable
from agentkit.pii.provider import wrap_provider
from agentkit.providers.fakes import FakeProvider
from agentkit.tools.registry import ToolRegistry

from ..pii.conftest import FakeDetector, FakeTokenMap


def _drain(queue: asyncio.Queue) -> list:
    out = []
    while not queue.empty():
        out.append(queue.get_nowait())
    return out


def _user(text: str) -> Message:
    return Message(
        id=new_id(MessageId),
        session_id=new_id(SessionId),
        role=MessageRole.USER,
        content=[TextBlock(text=text)],
        created_at=datetime.now(UTC),
    )


async def _run_with_raiser(exc: BaseException) -> tuple[list[Errored], dict]:
    """Run one turn whose first handler raises ``exc``; return the Errored events."""

    async def boom(ctx, deps):
        raise exc

    ctx = TurnContext.empty()
    ctx.event_queue = asyncio.Queue()
    loop = Loop(ctx=ctx, handlers={Phase.INTENT_GATE: boom})
    [ev async for ev in loop.run()]
    errored = [e for e in _drain(ctx.event_queue) if isinstance(e, Errored)]
    return errored, ctx.metadata["turn_error"]


# --- error_code_for ---------------------------------------------------------


def test_bare_exception_is_internal():
    assert error_code_for(RuntimeError("boom")) is ErrorCode.INTERNAL


def test_agentkit_base_defaults_to_internal():
    """An exception that says nothing must behave exactly as it did before."""
    assert error_code_for(AgentkitError("boom")) is ErrorCode.INTERNAL


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (ProviderError("upstream 500"), ErrorCode.PROVIDER_FAULT),
        (ToolError("dispatch failed"), ErrorCode.TOOL_FAULT),
        (ApprovalTimeout("expired"), ErrorCode.APPROVAL_TIMEOUT),
        (ZdrRouteUnavailable("no route"), ErrorCode.POLICY_REFUSED),
        (BlockedModelError("blocked"), ErrorCode.POLICY_REFUSED),
        # Deliberately INTERNAL: these are agentkit malfunctioning, not refusing.
        (ConfigurationError("bad config"), ErrorCode.INTERNAL),
        (CheckpointMissing("gone"), ErrorCode.INTERNAL),
        (StoreError("db down"), ErrorCode.INTERNAL),
        (InvalidPhaseTransition("a", "b"), ErrorCode.INTERNAL),
    ],
)
def test_each_exception_names_its_code(exc: Exception, expected: ErrorCode):
    assert error_code_for(exc) is expected


def test_instance_may_override_the_class_code():
    """One class can cover outcomes that differ — a 429 raised as ProviderError."""
    exc = ProviderError("429 from upstream")
    exc.code = ErrorCode.RATE_LIMITED
    assert error_code_for(exc) is ErrorCode.RATE_LIMITED
    # The class default is untouched for the next instance.
    assert error_code_for(ProviderError("500")) is ErrorCode.PROVIDER_FAULT


def test_a_consumer_exception_can_name_its_code_without_inheriting():
    """The hook is duck-typed: a consumer cannot subclass a class it does not own."""

    class ConsumerToolFailure(Exception):
        code = ErrorCode.TOOL_FAULT

    assert error_code_for(ConsumerToolFailure("nope")) is ErrorCode.TOOL_FAULT


def test_a_foreign_string_code_is_ignored_not_coerced():
    """``.code`` is a common attribute name and is usually a bare string.

    ``openai.APIStatusError.code`` is ``"not_found"``; ``ErrorCode("not_found")``
    would raise ValueError, on an error path, turning a reportable failure into
    an unreportable one. Only a real member is honoured.
    """

    class ForeignHttpError(Exception):
        code = "not_found"

    assert error_code_for(ForeignHttpError("404")) is ErrorCode.INTERNAL

    class ForeignNumericError(Exception):
        code = 500

    assert error_code_for(ForeignNumericError("500")) is ErrorCode.INTERNAL

    # A bare string equal to a member's *value* is still not a member.
    class LooksRight(Exception):
        code = "policy_refused"

    assert error_code_for(LooksRight("x")) is ErrorCode.INTERNAL


def test_every_declared_code_is_a_real_member():
    """A typo'd class attribute would silently ship a code nothing can match."""

    def subclasses(cls: type) -> list[type]:
        out = []
        for sub in cls.__subclasses__():
            out.append(sub)
            out.extend(subclasses(sub))
        return out

    for cls in [AgentkitError, *subclasses(AgentkitError)]:
        assert isinstance(cls.code, ErrorCode), f"{cls.__name__}.code is {cls.code!r}"


# --- the orchestrator funnel ------------------------------------------------


@pytest.mark.asyncio
async def test_handler_exception_carries_the_exception_code():
    errored, stored = await _run_with_raiser(ZdrRouteUnavailable("no ZDR route"))
    assert len(errored) == 1
    assert errored[0].code is ErrorCode.POLICY_REFUSED
    assert errored[0].recoverable is False
    # The persisted turn discriminates the same way the live event did, so a
    # conversation read back from history is not flattened to "internal".
    assert stored["code"] == ErrorCode.POLICY_REFUSED.value
    assert stored["type"] == "ZdrRouteUnavailable"


@pytest.mark.asyncio
async def test_an_unhandled_crash_is_still_internal():
    errored, stored = await _run_with_raiser(RuntimeError("handler exploded"))
    assert errored[0].code is ErrorCode.INTERNAL
    assert stored["code"] == ErrorCode.INTERNAL.value


@pytest.mark.asyncio
async def test_tool_and_provider_failures_reach_the_wire_distinguishably():
    tool, _ = await _run_with_raiser(ToolError("dispatch failed"))
    provider, _ = await _run_with_raiser(ProviderError("upstream 500"))
    assert tool[0].code is ErrorCode.TOOL_FAULT
    assert provider[0].code is ErrorCode.PROVIDER_FAULT


@pytest.mark.asyncio
async def test_no_handler_has_no_exception_and_stays_internal():
    """The one funnel path with nothing to ask. INTERNAL is what it means."""
    ctx = TurnContext.empty()
    ctx.event_queue = asyncio.Queue()
    loop = Loop(ctx=ctx, handlers={})
    [ev async for ev in loop.run()]
    errored = [e for e in _drain(ctx.event_queue) if isinstance(e, Errored)]
    assert errored[0].code is ErrorCode.INTERNAL
    assert ctx.metadata["turn_error"]["code"] == ErrorCode.INTERNAL.value


@pytest.mark.asyncio
async def test_illegal_transition_stays_internal():
    async def jump(ctx, deps):
        return Phase.TOOL_EXECUTING  # not reachable from INTENT_GATE

    ctx = TurnContext.empty()
    ctx.event_queue = asyncio.Queue()
    loop = Loop(ctx=ctx, handlers={Phase.INTENT_GATE: jump})
    [ev async for ev in loop.run()]
    errored = [e for e in _drain(ctx.event_queue) if isinstance(e, Errored)]
    assert errored[0].code is ErrorCode.INTERNAL


@pytest.mark.asyncio
async def test_message_is_unchanged_so_existing_log_consumers_still_match():
    """The fix adds a code; it does not reword the diagnostic field."""
    errored, _ = await _run_with_raiser(ZdrRouteUnavailable("no ZDR route"))
    assert errored[0].message == "intent_gate: ZdrRouteUnavailable: no ZDR route"


# --- the seam that reported the bug -----------------------------------------


@pytest.mark.asyncio
async def test_a_firewall_refusal_reaches_the_consumer_as_policy_refused():
    """Provider -> firewall -> streaming handler -> orchestrator -> wire.

    This is the reported path, and it is the only test here that would have
    caught the bug: every stage is the real one. The firewall raises because the
    routed model has no zero-data-retention endpoint, and the consumer must be
    able to say so instead of showing a generic failure.
    """
    inner = FakeProvider().script(
        FakeProvider.error("no_compliant_provider", "no ZDR endpoints found")
    )
    firewall = Firewall(detector=FakeDetector(), policy=PiiPolicy())
    tmap = FakeTokenMap()
    provider = wrap_provider(inner, firewall, tmap_resolver=lambda req: tmap)

    ctx = TurnContext.empty()
    ctx.add_message(_user("ring Kari Nordmann"))
    ctx.event_queue = asyncio.Queue()
    loop = Loop(
        ctx=ctx,
        handlers={Phase.STREAMING: handle_streaming},
        starting_phase=Phase.STREAMING,
        deps={
            "provider": provider,
            "message_builder": MessageBuilder(model="m", max_tokens=128),
            "registry": ToolRegistry(),
            "system_blocks": [],
            "success_claim": None,
        },
    )
    [ev async for ev in loop.run()]

    errored = [e for e in _drain(ctx.event_queue) if isinstance(e, Errored)]
    assert len(errored) == 1, f"expected one Errored, got {errored}"
    assert errored[0].code is ErrorCode.POLICY_REFUSED
    assert errored[0].recoverable is False


# --- the wire ---------------------------------------------------------------


def test_the_new_code_round_trips_through_the_event_adapter():
    ctx = TurnContext.empty()
    evt = Errored(
        event_id=new_id(EventId),
        session_id=ctx.session_id,
        turn_id=ctx.turn_id,
        ts=datetime.now(UTC),
        sequence=0,
        code=ErrorCode.POLICY_REFUSED,
        message="refused",
        recoverable=False,
    )
    back = EVENT_ADAPTER.validate_json(evt.model_dump_json())
    assert isinstance(back, Errored)
    assert back.code is ErrorCode.POLICY_REFUSED
    assert '"code":"policy_refused"' in evt.model_dump_json()
