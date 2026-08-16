"""K12: a genuine write from an earlier turn must not be flagged as
``fabricated_tool`` once compaction has summarized that turn away.

Rule 1 (fabricated_tool, see finalize_validator.py) matches
``actions_performed[].tool`` against successful writes it finds by walking
``ctx.history`` for ``ToolUseBlock``/``ToolResultBlock`` pairs
(``guards.finalize._ctx_to_summaries``). That works for any turn still
present in the loaded window. ``compact_history`` (compaction.py) is the one
thing in this codebase that removes a turn from that window on purpose,
replacing it with a single prose summary message — which has no
``ToolUseBlock`` for the validator to find.

So a model that genuinely called a write tool, had that turn compacted away,
and later (truthfully) lists that tool in ``actions_performed`` gets rejected
as if it had invented the call. This reproduces that at the real seam:
``StructuralFinalizeValidator.validate``, over a history built by the real
``compact_history`` — not a hand-rolled substitute for it.
"""

from datetime import UTC, datetime

import pytest

from agentkit._content import TextBlock, ToolResultBlock, ToolUseBlock
from agentkit._ids import MessageId, OwnerId, SessionId, new_id
from agentkit._messages import Message, MessageMetadata, MessageRole
from agentkit.compaction import compact_history
from agentkit.guards.finalize import StructuralFinalizeValidator
from agentkit.loop.context import TurnContext
from agentkit.store.fakes import FakeSessionStore
from agentkit.tools.spec import ToolCall


def _msg(sid: SessionId, role: MessageRole, *content) -> Message:
    return Message(
        id=new_id(MessageId),
        session_id=sid,
        role=role,
        content=list(content),
        metadata=MessageMetadata(),
        created_at=datetime.now(UTC),
    )


async def _seed(store: FakeSessionStore, sid: SessionId, messages: list[Message]) -> None:
    await store.create(sid, OwnerId("u:1"))
    for m in messages:
        await store.append_message(sid, m)


async def _fake_summarizer(prefix: list[Message]) -> str:
    return f"summarized {len(prefix)} messages, including a matrix notification"


def _make_finalize_call(tool: str) -> ToolCall:
    return ToolCall(
        id="finalize-1",
        name="kit.finalize",
        arguments={
            "status": "done",
            "intent_kind": "action",
            "actions_performed": [
                {"tool": tool, "target": None, "description": "sent the matrix notification"}
            ],
        },
    )


async def _build_post_compaction_ctx(sid: SessionId, store: FakeSessionStore) -> TurnContext:
    """Genuine notify_matrix write in turn 0, then enough padding turns that
    compaction summarizes turn 0 away and only the padding tail survives."""
    messages = [
        _msg(sid, MessageRole.USER, TextBlock(text="send the matrix notification")),
        _msg(sid, MessageRole.ASSISTANT, ToolUseBlock(id="u1", name="notify_matrix", arguments={})),
        _msg(
            sid,
            MessageRole.TOOL,
            ToolResultBlock(tool_use_id="u1", content=[TextBlock(text="sent")], is_error=False),
        ),
        _msg(sid, MessageRole.ASSISTANT, TextBlock(text="Sent the notification.")),
    ]
    for i in range(10):
        messages.append(_msg(sid, MessageRole.USER, TextBlock(text=f"chat {i}")))
        messages.append(_msg(sid, MessageRole.ASSISTANT, TextBlock(text=f"reply {i}")))
    await _seed(store, sid, messages)

    result = await compact_history(store, sid, _fake_summarizer, keep_recent=6, min_messages=12)
    assert result.compacted is True

    history = await store.list_messages(sid)
    # Sanity: the compacted history no longer carries a notify_matrix ToolUseBlock.
    assert not any(
        isinstance(b, ToolUseBlock) and b.name == "notify_matrix"
        for m in history
        for b in m.content
    )

    history.append(_msg(sid, MessageRole.USER, TextBlock(text="thanks, all good?")))

    ctx = TurnContext.empty()
    ctx.add_messages(history)
    return ctx


@pytest.mark.asyncio
async def test_genuine_write_from_compacted_turn_is_not_fabricated():
    sid = new_id(SessionId)
    store = FakeSessionStore()
    ctx = await _build_post_compaction_ctx(sid, store)

    validator = StructuralFinalizeValidator()
    verdict = await validator.validate(_make_finalize_call("notify_matrix"), ctx)

    assert verdict.accept, verdict.feedback


@pytest.mark.asyncio
async def test_a_tool_never_actually_called_is_still_fabricated_after_compaction():
    """The fix must not turn Rule 1 into a rubber stamp: a tool that never
    appears anywhere in the pre-compaction transcript is still fabricated."""
    sid = new_id(SessionId)
    store = FakeSessionStore()
    ctx = await _build_post_compaction_ctx(sid, store)

    validator = StructuralFinalizeValidator()
    verdict = await validator.validate(_make_finalize_call("delete_everything"), ctx)

    assert not verdict.accept
    assert "fabricated_tool" in (verdict.feedback or "")
