"""fresh_child_context: taint latches downward, and so does its provenance.

A parent turn that already ingested untrusted content must not be able to
launder it through a fresh child whose ``tainted`` flag starts clean — but a
child that only inherits the boolean and not ``taint_sources`` knows it is
tainted without any record of why, which defeats provenance reporting on the
child's own ApprovalNeeded cards.
"""

from agentkit.guards.taint import TaintSource
from agentkit.loop.context import TurnContext
from agentkit.subagents.isolation import fresh_child_context


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
