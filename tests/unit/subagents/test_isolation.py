"""fresh_child_context: taint latches downward, and so does its provenance.

A parent turn that already ingested untrusted content must not be able to
launder it through a fresh child whose ``tainted`` flag starts clean — but a
child that only inherits the boolean and not ``taint_sources`` knows it is
tainted without any record of why, which defeats provenance reporting on the
child's own ApprovalNeeded cards.
"""

import asyncio

import pytest

from agentkit.events import ApprovalNeeded
from agentkit.guards.taint import TaintSource
from agentkit.loop.context import TurnContext
from agentkit.loop.handlers.approval_wait import handle_approval_wait
from agentkit.subagents.isolation import fresh_child_context
from agentkit.tools.registry import ToolRegistry


def _tainted_parent(*sources: TaintSource) -> TurnContext:
    parent = TurnContext.empty()
    parent.tainted = True
    parent.taint_sources = list(sources)
    return parent


def test_child_inherits_taint_sources_content():
    s1 = TaintSource(call_id="c0", tool_name="web.fetch", kind="untrusted")
    s2 = TaintSource(call_id="c1", tool_name="email.read", kind="untrusted")
    parent = _tainted_parent(s1, s2)

    child = fresh_child_context(parent, prompt="do the thing")

    assert child.tainted is True
    assert child.taint_sources == [s1, s2]


def test_clean_parent_produces_a_clean_child_with_no_sources():
    parent = TurnContext.empty()
    assert parent.tainted is False
    assert parent.taint_sources == []

    child = fresh_child_context(parent, prompt="do the thing")

    assert child.tainted is False
    assert child.taint_sources == []


def test_mutating_the_childs_taint_sources_does_not_corrupt_the_parents():
    """taint_sources is a mutable list; sharing the reference would let a
    child's own ingestion (or a bug) silently rewrite the parent's record."""
    s1 = TaintSource(call_id="c0", tool_name="web.fetch", kind="untrusted")
    parent = _tainted_parent(s1)

    child = fresh_child_context(parent, prompt="do the thing")
    assert child.taint_sources is not parent.taint_sources  # not the same list object

    child.taint_sources.append(TaintSource(call_id="c9", tool_name="child.tool", kind="untrusted"))

    assert parent.taint_sources == [s1]
    assert child.taint_sources == [
        s1,
        TaintSource(call_id="c9", tool_name="child.tool", kind="untrusted"),
    ]


@pytest.mark.asyncio
async def test_a_childs_approval_card_names_what_tainted_its_parent():
    """The seam, rather than the field copy the three tests above assert on.

    What ``taint_sources`` is *for* is ``ApprovalNeeded.taint`` — the list a
    human approver reads to see whose text the model had ingested before it
    proposed this action. Nothing else consumes it: the taint gate
    (``RiskBasedTaintPolicy.denial_reason``) branches on the ``tainted``
    boolean alone, which is why the omission never showed up as a permission
    failure and could sit there looking harmless. It cost provenance, and
    provenance is only observable here.

    A subagent does reach this handler: ``APPROVAL_WAIT`` is in
    ``_CHILD_HANDLERS``, and the child's events are forwarded onto the parent's
    stream before ``SubagentDispatcher._check_no_pending_approvals`` raises. So
    a deployment that supplies its own approval gate — one the dispatcher did
    not wrap in ``NestedApprovalGate`` — renders a real card from a child
    context. Pre-fix that card said "this turn is tainted" and listed nothing.
    """
    ingested = TaintSource(call_id="c0", tool_name="web.fetch", kind="untrusted")
    parent = _tainted_parent(ingested)

    child = fresh_child_context(parent, prompt="summarise that page")
    child.event_queue = asyncio.Queue()
    child.metadata["pending_user_approvals"] = [
        {"id": "c1", "name": "vault.delete", "arguments": {"id": "42"}}
    ]

    await handle_approval_wait(child, {"registry": ToolRegistry()})

    card = child.event_queue.get_nowait()
    assert isinstance(card, ApprovalNeeded)
    assert card.taint == [ingested], "the card must name what the model read, not just that it read"
