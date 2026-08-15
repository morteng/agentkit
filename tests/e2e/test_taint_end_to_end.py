"""Taint/provenance, end to end through a real AgentSession.

The unit tests under ``tests/unit/guards/test_taint.py`` and
``tests/unit/tools/test_registry_gates.py`` pin the policy and the registry
gate in isolation. This module pins the property the feature actually claims,
across every layer that has to cooperate for it to hold — dispatcher, registry,
loop phases, checkpoint serialization and ``AgentSession``:

    once a turn has read untrusted content, that turn cannot write.

Each layer was built by a different pass, so an isolated-component green tells
us little; a break anywhere in the chain (a dispatcher that forgets to mark, a
checkpoint that drops the flag on resume) shows up here and only here.
"""

import asyncio

import pytest

from agentkit import AgentConfig, AgentSession, Provenance
from agentkit._ids import OwnerId
from agentkit.events import ApprovalNeeded, ToolCallResult, TurnEnded, TurnEndReason
from agentkit.guards.approval import RiskBasedApprovalGate
from agentkit.guards.taint import TAINT_DENIAL_CODE
from agentkit.providers.fakes import FakeProvider
from agentkit.store.fakes import FakeCheckpointStore, FakeSessionStore
from agentkit.tools.builtin import DEFAULT_BUILTINS
from agentkit.tools.registry import ToolRegistry
from agentkit.tools.spec import (
    ApprovalPolicy,
    ContentBlockOut,
    RiskLevel,
    SideEffects,
    ToolResult,
    ToolSpec,
)

pytestmark = pytest.mark.e2e


def _spec(name: str, risk: RiskLevel, approval: ApprovalPolicy) -> ToolSpec:
    return ToolSpec(
        name=name,
        description=name,
        parameters={"type": "object"},
        returns=None,
        risk=risk,
        idempotent=False,
        side_effects=SideEffects.LOCAL,
        requires_approval=approval,
        cache_ttl_seconds=None,
        timeout_seconds=10.0,
    )


WEB_FETCH = _spec("web.fetch", RiskLevel.READ, ApprovalPolicy.NEVER)
NOTES_READ = _spec("notes.read", RiskLevel.READ, ApprovalPolicy.NEVER)
# LOW_WRITE, ApprovalPolicy.NEVER: auto-approved, so the call reaches
# ``registry.invoke``. If the taint guard were absent this tool would run —
# which is exactly what makes it the right probe.
DOCS_WRITE = _spec("docs.write", RiskLevel.LOW_WRITE, ApprovalPolicy.NEVER)
DEVICES_DELETE = _spec("devices.delete", RiskLevel.HIGH_WRITE, ApprovalPolicy.BY_RISK)

_FINALIZE_ARGS = {
    "status": "done",
    "intent_kind": "answer",
    "summary": "Done.",
    "answer_evidence": "general_knowledge",
}


class _Tools:
    """Handlers that record every execution, so 'never ran' is assertable."""

    def __init__(self) -> None:
        self.executed: list[str] = []

    def _ok(self, name: str, ctx, *, provenance: Provenance) -> ToolResult:
        self.executed.append(name)
        return ToolResult(
            call_id=ctx.call_id,
            status="ok",
            content=[ContentBlockOut(type="text", text=f"{name} ran")],
            error=None,
            duration_ms=1,
            cached=False,
            provenance=provenance,
        )

    async def web_fetch(self, args, ctx) -> ToolResult:
        # The one tool that ingests third-party bytes. Marking the result is
        # the tool author's opt-in — nothing in agentkit infers it.
        return self._ok("web.fetch", ctx, provenance=Provenance.UNTRUSTED)

    async def notes_read(self, args, ctx) -> ToolResult:
        return self._ok("notes.read", ctx, provenance=Provenance.SYSTEM)

    async def docs_write(self, args, ctx) -> ToolResult:
        return self._ok("docs.write", ctx, provenance=Provenance.SYSTEM)

    async def devices_delete(self, args, ctx) -> ToolResult:
        return self._ok("devices.delete", ctx, provenance=Provenance.SYSTEM)


def _session(provider: FakeProvider) -> tuple[AgentSession, _Tools]:
    config = AgentConfig()
    config.guards.approval = RiskBasedApprovalGate()
    config.stores.session = FakeSessionStore()
    config.stores.checkpoint = FakeCheckpointStore()

    registry = ToolRegistry()
    for spec, handler in DEFAULT_BUILTINS:
        registry.register_builtin(spec, handler)

    tools = _Tools()
    registry.register_builtin(WEB_FETCH, tools.web_fetch)
    registry.register_builtin(NOTES_READ, tools.notes_read)
    registry.register_builtin(DOCS_WRITE, tools.docs_write)
    registry.register_builtin(DEVICES_DELETE, tools.devices_delete)

    session = AgentSession(
        owner=OwnerId("u:1"),
        config=config,
        provider=provider,
        registry=registry,
        model="m",
    )
    return session, tools


async def _collect(stream) -> list:
    return [ev async for ev in stream]


def _results(events: list) -> dict[str, ToolCallResult]:
    """Tool results keyed by the tool that produced them.

    ``ToolCallResult`` carries no tool name, so match on the recorded content /
    denial text, which both name the tool.
    """
    out: dict[str, ToolCallResult] = {}
    for ev in events:
        if not isinstance(ev, ToolCallResult):
            continue
        text = " ".join(b.text or "" for b in ev.content)
        for name in ("web.fetch", "notes.read", "docs.write", "devices.delete"):
            if name in text:
                out[name] = ev
    return out


@pytest.mark.asyncio
async def test_untrusted_read_disables_writes_for_the_rest_of_the_turn():
    """One untrusted READ taints the turn; the write that follows is refused."""
    provider = FakeProvider().script(
        FakeProvider.tool_call("web.fetch", {}),
        # Both in one assistant message: proves the denial is per-call and not
        # a blanket stop on the turn — the read in the same batch still runs.
        FakeProvider.tool_calls([("docs.write", {}), ("notes.read", {})]),
        FakeProvider.tool_call("kit.finalize", _FINALIZE_ARGS),
    )
    session, tools = _session(provider)

    async with session.run("summarise example.com and save a note") as stream:
        events = await _collect(stream)

    assert isinstance(events[-1], TurnEnded)
    assert events[-1].reason is TurnEndReason.COMPLETED

    # The write never reached its handler.
    assert "docs.write" not in tools.executed
    # The read did — a tainted turn can still finish answering the question
    # that pulled the untrusted content in.
    assert tools.executed == ["web.fetch", "notes.read"]

    results = _results(events)
    denial = results["docs.write"]
    assert denial.status == "denied"
    assert denial.error is not None
    assert denial.error.code == TAINT_DENIAL_CODE
    # Not retryable inside this turn: the model must not burn iterations on it.
    assert denial.error.retryable is False
    # The message tells the model what to do about it, in words it can relay.
    assert "new turn" in denial.error.message
    assert results["notes.read"].status == "ok"


@pytest.mark.asyncio
async def test_a_fresh_turn_starts_clean():
    """Taint is per-turn: the next turn on the same session may write again."""
    provider = FakeProvider().script(
        # Turn 1: taints, then tries to write.
        FakeProvider.tool_call("web.fetch", {}),
        FakeProvider.tool_call("docs.write", {}),
        FakeProvider.tool_call("kit.finalize", _FINALIZE_ARGS),
        # Turn 2: the user restates the action, as the denial told them to.
        FakeProvider.tool_call("docs.write", {}),
        FakeProvider.tool_call("kit.finalize", _FINALIZE_ARGS),
    )
    session, tools = _session(provider)

    async with session.run("summarise example.com and save a note") as stream:
        first = await _collect(stream)
    assert _results(first)["docs.write"].status == "denied"
    assert tools.executed == ["web.fetch"]

    async with session.run("save the note") as stream:
        second = await _collect(stream)

    assert isinstance(second[-1], TurnEnded)
    assert second[-1].reason is TurnEndReason.COMPLETED
    assert _results(second)["docs.write"].status == "ok"
    assert tools.executed == ["web.fetch", "docs.write"]


@pytest.mark.asyncio
async def test_taint_survives_an_approval_suspend_and_resume():
    """The flag round-trips through the checkpoint.

    A suspended turn is rebuilt field by field from its checkpoint payload. If
    ``tainted`` did not survive that, every approval suspend would silently
    launder untrusted content: the turn would come back clean and re-enable
    precisely the writes the guard had just blocked.

    Note the deliberate consequence pinned at the end: the guard is *below*
    approval, so a human "approve" does not lift it. The user is asked, the
    card carries the taint sources (``ApprovalNeeded.taint``), and the call is
    still refused. Fail-closed, but see the release notes — whether an explicit
    human grant should override taint is a policy question, not an accident.
    """
    provider = FakeProvider().script(
        FakeProvider.tool_call("web.fetch", {}),
        FakeProvider.tool_call("devices.delete", {"id": "x"}),
        FakeProvider.tool_call("kit.finalize", _FINALIZE_ARGS),
    )
    session, tools = _session(provider)

    needed: ApprovalNeeded | None = None
    async with session.run("read example.com then delete device x") as stream:
        async for ev in stream:
            if isinstance(ev, ApprovalNeeded):
                needed = ev
            elif isinstance(ev, TurnEnded):
                assert ev.reason is TurnEndReason.AWAITING_APPROVAL

    assert needed is not None
    # The card tells the approver the model had read untrusted content first.
    assert [s.tool_name for s in needed.taint] == ["web.fetch"]

    async with session.resume_with_approval(
        needed.turn_id, needed.call_id, decision="approve"
    ) as stream:
        resumed = await _collect(stream)

    result = _results(resumed)["devices.delete"]
    assert result.status == "denied"
    assert result.error is not None
    assert result.error.code == TAINT_DENIAL_CODE
    assert "devices.delete" not in tools.executed


@pytest.mark.asyncio
async def test_a_trusted_tool_result_does_not_taint_anything():
    """The control: without an UNTRUSTED result the guard must never fire.

    Provenance defaults to ``SYSTEM``, so every pre-existing tool keeps working
    exactly as before. A guard that also denied here would be a regression for
    every consumer, not a security control.
    """
    provider = FakeProvider().script(
        FakeProvider.tool_call("notes.read", {}),
        FakeProvider.tool_call("docs.write", {}),
        FakeProvider.tool_call("kit.finalize", _FINALIZE_ARGS),
    )
    session, tools = _session(provider)

    async with session.run("read the note then save it") as stream:
        events = await _collect(stream)

    assert tools.executed == ["notes.read", "docs.write"]
    assert _results(events)["docs.write"].status == "ok"
    assert isinstance(events[-1], TurnEnded)


@pytest.mark.asyncio
async def test_concurrent_sessions_do_not_share_taint():
    """Taint lives on the TurnContext, not on the registry or any global.

    Two sessions run interleaved on the same event loop; only one of them reads
    untrusted content. Module-level or registry-level state would leak the
    denial into the clean session.
    """
    dirty_provider = FakeProvider().script(
        FakeProvider.tool_call("web.fetch", {}),
        FakeProvider.tool_call("docs.write", {}),
        FakeProvider.tool_call("kit.finalize", _FINALIZE_ARGS),
    )
    clean_provider = FakeProvider().script(
        FakeProvider.tool_call("docs.write", {}),
        FakeProvider.tool_call("kit.finalize", _FINALIZE_ARGS),
    )
    dirty_session, _ = _session(dirty_provider)
    clean_session, clean_tools = _session(clean_provider)

    async def run(session: AgentSession, text: str) -> list:
        async with session.run(text) as stream:
            return await _collect(stream)

    dirty_events, clean_events = await asyncio.gather(
        run(dirty_session, "read the web then write"),
        run(clean_session, "just write"),
    )

    assert _results(dirty_events)["docs.write"].status == "denied"
    assert _results(clean_events)["docs.write"].status == "ok"
    assert clean_tools.executed == ["docs.write"]
