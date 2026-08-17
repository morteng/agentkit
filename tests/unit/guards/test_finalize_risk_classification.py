"""The write/read split must come from the registry, not from tool names.

Regression for a live incident. A household assistant ran sixteen
``torrent_search`` calls, never called ``torrent_add``, and finalized with
``intent_kind="action"``. Rule 1 (``fabricated_tool``) exists to catch exactly
that, and it passed — because ``_is_default_write`` classifies a name as a read
only when it *starts with* one of ``search``/``get_``/``list_``/..., and
``torrent_search`` starts with ``torrent_``. Every tool in that consumer's
registry is named ``noun_verb``, so every tool classified as a write, and a
turn that only read looked like a turn full of successful writes.

The bug is not that the heuristic is wrong; it is that a heuristic over *names*
encodes a naming convention, and the registry already holds the answer.
"""

from datetime import UTC, datetime

import pytest

from agentkit._content import ToolUseBlock
from agentkit._ids import MessageId, SessionId, new_id
from agentkit._messages import Message, MessageRole
from agentkit.guards.finalize import StructuralFinalizeValidator
from agentkit.loop.context import TurnContext
from agentkit.tools.spec import ApprovalPolicy, RiskLevel, SideEffects, ToolCall, ToolSpec


def _spec(name: str, risk: RiskLevel) -> ToolSpec:
    """A real ToolSpec, not a stub.

    Built through the genuine constructor so a field this test relies on
    changing shape fails here rather than passing against a mock that the
    library would have rejected.
    """
    return ToolSpec(
        name=name,
        description=f"{name} (test)",
        parameters={"type": "object", "properties": {}},
        risk=risk,
        idempotent=risk is RiskLevel.READ,
        side_effects=SideEffects.NONE
        if risk is RiskLevel.READ
        else SideEffects.EXTERNAL_REVERSIBLE,
        requires_approval=ApprovalPolicy.NEVER,
        cache_ttl_seconds=None,
        timeout_seconds=30.0,
    )


class _Registry:
    """Minimal stand-in for ToolRegistry exposing only ``spec_for``.

    Keyed by the qualified name, which is how tools are actually registered —
    the validator has to try both forms to resolve a bare name out of the call
    log, and keying only the qualified form is what makes that a real test
    rather than one that passes either way.
    """

    def __init__(self, specs: dict[str, RiskLevel]) -> None:
        self._specs = {name: _spec(name, risk) for name, risk in specs.items()}

    def spec_for(self, name: str) -> ToolSpec | None:
        return self._specs.get(name)


# The shape of the real registry at the time of the incident: reads and writes
# alike named `noun_verb`, which the name heuristic cannot tell apart.
_GULDEN_LIKE = _Registry(
    {
        "torrent.torrent_search": RiskLevel.READ,
        "media.calibre_search": RiskLevel.READ,
        "ops.disk_usage": RiskLevel.READ,
        "torrent_admin.torrent_add": RiskLevel.HIGH_WRITE,
        "provisioning.user_add": RiskLevel.HIGH_WRITE,
    }
)


def _msg(role: MessageRole, content: list) -> Message:
    return Message(
        id=new_id(MessageId),
        session_id=new_id(SessionId),
        role=role,
        content=content,
        created_at=datetime.now(UTC),
    )


def _ctx_with_calls(*names: str) -> TurnContext:
    ctx = TurnContext.empty()
    ctx.add_message(_msg(MessageRole.USER, []))
    ctx.add_message(
        _msg(
            MessageRole.ASSISTANT,
            [ToolUseBlock(id=f"u{i}", name=n, arguments={}) for i, n in enumerate(names)],
        )
    )
    return ctx


def _finalize(tool: str) -> ToolCall:
    return ToolCall(
        id="finalize-1",
        name="finalize_response",
        arguments={
            "status": "done",
            "intent_kind": "action",
            "actions_performed": [{"tool": tool, "target": "the queue", "description": "added it"}],
        },
    )


@pytest.mark.asyncio
async def test_a_read_tool_cannot_be_reported_as_an_action_performed():
    """The incident, reduced: searched only, then claimed an action.

    Goes red without the risk lookup — ``torrent_search`` matches no read
    prefix, so it counts as a successful write and Rule 1 is satisfied.
    """
    validator = StructuralFinalizeValidator(_GULDEN_LIKE)
    verdict = await validator.validate(
        _finalize("torrent_search"),
        _ctx_with_calls("torrent.torrent_search", "torrent.torrent_search"),
    )
    assert not verdict.accept
    assert verdict.feedback is not None
    assert "fabricated_tool" in verdict.feedback


@pytest.mark.asyncio
async def test_a_genuine_write_is_still_accepted():
    """Control. Without this, the test above passes in a build that rejects
    every envelope, which would be a worse bug than the one being fixed."""
    validator = StructuralFinalizeValidator(_GULDEN_LIKE)
    verdict = await validator.validate(
        _finalize("torrent_add"),
        _ctx_with_calls("torrent.torrent_search", "torrent_admin.torrent_add"),
    )
    assert verdict.accept, verdict.feedback


@pytest.mark.asyncio
async def test_every_read_in_the_registry_is_classified_as_a_read():
    """Drive every read tool, not a sample.

    The original defect was uniform across the registry — testing one name
    would have proven only that one name was fixed, and the whole point is
    that the naming convention makes them fail together.
    """
    validator = StructuralFinalizeValidator(_GULDEN_LIKE)
    reads = ["torrent.torrent_search", "media.calibre_search", "ops.disk_usage"]
    for qualified in reads:
        bare = qualified.split(".", 1)[-1]
        verdict = await validator.validate(_finalize(bare), _ctx_with_calls(qualified))
        assert not verdict.accept, f"{bare} was accepted as a performed action"
        assert "fabricated_tool" in (verdict.feedback or "")


@pytest.mark.asyncio
async def test_an_unknown_name_still_fails_closed_to_the_heuristic():
    """A name no registry can resolve keeps the old behaviour.

    Calls carried across a compaction boundary lose their spec, and a caller
    may have no registry at all. Both must stay usable, and an unresolvable
    name must still count as a write rather than silently becoming a read —
    fail-closed is the property being preserved.
    """
    validator = StructuralFinalizeValidator(_GULDEN_LIKE)
    verdict = await validator.validate(_finalize("patch_content"), _ctx_with_calls("patch_content"))
    assert verdict.accept, verdict.feedback


@pytest.mark.asyncio
async def test_without_a_registry_the_validator_still_works():
    """Every existing construction site passes no registry. They must not break."""
    validator = StructuralFinalizeValidator()
    verdict = await validator.validate(_finalize("patch_content"), _ctx_with_calls("patch_content"))
    assert verdict.accept, verdict.feedback
