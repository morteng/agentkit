from datetime import UTC, datetime

import pytest

from agentkit._content import TextBlock, ToolResultBlock, ToolUseBlock
from agentkit._ids import MessageId, OwnerId, SessionId, new_id
from agentkit._messages import (
    INJECTED_CORRECTION_ANNOTATION,
    Message,
    MessageMetadata,
    MessageRole,
)
from agentkit.compaction import COMPACTION_SUMMARY_ANNOTATION, compact_history
from agentkit.store.fakes import FakeSessionStore


def _msg(sid: SessionId, role: MessageRole, *content, annotations=None) -> Message:
    return Message(
        id=new_id(MessageId),
        session_id=sid,
        role=role,
        content=list(content),
        metadata=MessageMetadata(annotations=annotations or {}),
        created_at=datetime.now(UTC),
    )


def _user(sid: SessionId, text: str) -> Message:
    return _msg(sid, MessageRole.USER, TextBlock(text=text))


def _assistant(sid: SessionId, text: str) -> Message:
    return _msg(sid, MessageRole.ASSISTANT, TextBlock(text=text))


async def _seed(store: FakeSessionStore, sid: SessionId, messages: list[Message]) -> None:
    await store.create(sid, OwnerId("u:1"))
    for m in messages:
        await store.append_message(sid, m)


async def _fake_summarizer(prefix: list[Message]) -> str:
    return f"summarized {len(prefix)} messages"


@pytest.mark.asyncio
async def test_min_messages_no_op_leaves_store_untouched():
    store = FakeSessionStore()
    sid = new_id(SessionId)
    messages = [_user(sid, f"turn {i}") for i in range(5)]
    await _seed(store, sid, messages)

    called = False

    async def summarizer(prefix: list[Message]) -> str:
        nonlocal called
        called = True
        return "should not be called"

    result = await compact_history(store, sid, summarizer, keep_recent=2, min_messages=12)

    assert result.compacted is False
    assert result.summarized_count == 0
    assert result.kept_count == 5
    assert result.summary == ""
    assert called is False
    assert await store.list_messages(sid) == messages


@pytest.mark.asyncio
async def test_naive_cut_already_safe_keeps_trailing_user_message():
    store = FakeSessionStore()
    sid = new_id(SessionId)
    # 12 plain user/assistant turns; naive cut at len-keep_recent=8 lands on
    # a USER message already, so no walking back is needed.
    messages = []
    for i in range(6):
        messages.append(_user(sid, f"user {i}"))
        messages.append(_assistant(sid, f"assistant {i}"))
    await _seed(store, sid, messages)
    assert messages[8].role is MessageRole.USER  # sanity: naive cut is safe

    result = await compact_history(store, sid, _fake_summarizer, keep_recent=4, min_messages=12)

    assert result.compacted is True
    assert result.summarized_count == 8
    assert result.kept_count == 4
    assert result.summary == "summarized 8 messages"

    new_history = await store.list_messages(sid)
    assert len(new_history) == 5  # 1 summary + 4 kept
    assert new_history[0].role is MessageRole.USER
    assert new_history[0].metadata.annotations.get(COMPACTION_SUMMARY_ANNOTATION) is True
    assert "conversation summary" in new_history[0].content[0].text  # type: ignore[union-attr]
    assert "summarized 8 messages" in new_history[0].content[0].text  # type: ignore[union-attr]
    # kept tail matches the original trailing messages exactly
    assert new_history[1:] == messages[8:]


@pytest.mark.asyncio
async def test_tool_pair_spanning_naive_cut_forces_earlier_cut():
    store = FakeSessionStore()
    sid = new_id(SessionId)
    # A pathological interleaving: a tool_use is issued (index 3), then a
    # USER message interjects (index 4, would be the naive cut) before the
    # matching tool_result arrives (index 5). Index 4 passes the "is this a
    # USER message" check on its own, but the span check must still reject
    # it — cutting there would orphan the tool_result at index 5 into the
    # kept tail with no preceding tool_use. The walk-back must continue
    # past both the TOOL-role message at 3... (role-unsafe) down to the
    # nearest boundary where NEITHER check trips: index 2.
    messages = [
        _user(sid, "u0"),  # 0
        _assistant(sid, "a0"),  # 1
        _user(sid, "u1"),  # 2 <- expected safe cut
        _msg(
            sid, MessageRole.ASSISTANT, ToolUseBlock(id="call-1", name="search", arguments={})
        ),  # 3
        _user(sid, "u2 (interjection)"),  # 4 <- naive cut; USER but tool_result pending
        _msg(  # 5
            sid,
            MessageRole.TOOL,
            ToolResultBlock(tool_use_id="call-1", content=[TextBlock(text="result")]),
        ),
        _assistant(sid, "a2"),  # 6
        _user(sid, "u3"),  # 7
        _assistant(sid, "a3"),  # 8
    ]
    await _seed(store, sid, messages)

    # naive_cut = len(9) - keep_recent(5) = 4.
    result = await compact_history(store, sid, _fake_summarizer, keep_recent=5, min_messages=9)

    assert result.compacted is True
    assert result.summarized_count == 2  # messages[:2] -> u0, a0
    assert result.kept_count == 7  # messages[2:]
    new_history = await store.list_messages(sid)
    assert len(new_history) == 8
    assert new_history[1:] == messages[2:]


@pytest.mark.asyncio
async def test_tool_pair_forces_cut_to_earlier_safe_user_boundary():
    store = FakeSessionStore()
    sid = new_id(SessionId)
    messages = [
        _user(sid, "u0"),  # 0
        _assistant(sid, "a0"),  # 1
        _user(sid, "u1"),  # 2 <- expected safe cut
        _msg(
            sid, MessageRole.ASSISTANT, ToolUseBlock(id="call-1", name="search", arguments={})
        ),  # 3
        _msg(  # 4 <- naive cut (len 10 - keep_recent 6 = 4); TOOL role, unsafe
            sid,
            MessageRole.TOOL,
            ToolResultBlock(tool_use_id="call-1", content=[TextBlock(text="result")]),
        ),
        _assistant(sid, "a1"),  # 5
        _user(sid, "u2"),  # 6
        _assistant(sid, "a2"),  # 7
        _user(sid, "u3"),  # 8
        _assistant(sid, "a3"),  # 9
    ]
    await _seed(store, sid, messages)

    result = await compact_history(store, sid, _fake_summarizer, keep_recent=6, min_messages=10)

    assert result.compacted is True
    assert result.summarized_count == 2  # messages[:2] -> u0, a0
    assert result.kept_count == 8  # messages[2:]
    new_history = await store.list_messages(sid)
    assert len(new_history) == 9
    assert new_history[1:] == messages[2:]


@pytest.mark.asyncio
async def test_injected_correction_user_message_is_not_a_safe_boundary():
    store = FakeSessionStore()
    sid = new_id(SessionId)
    messages = [
        _user(sid, "u0"),  # 0
        _assistant(sid, "a0"),  # 1
        _user(sid, "u1"),  # 2 <- expected safe cut (genuine turn boundary)
        _assistant(sid, "a1"),  # 3
        _msg(  # 4 <- naive cut point but annotated as injected correction
            sid,
            MessageRole.USER,
            TextBlock(text="please call finalize_response"),
            annotations={INJECTED_CORRECTION_ANNOTATION: True},
        ),
        _assistant(sid, "a2"),  # 5
        _user(sid, "u2"),  # 6
        _assistant(sid, "a3"),  # 7
    ]
    await _seed(store, sid, messages)

    result = await compact_history(store, sid, _fake_summarizer, keep_recent=4, min_messages=8)

    assert result.compacted is True
    assert result.summarized_count == 2
    assert result.kept_count == 6
    new_history = await store.list_messages(sid)
    assert new_history[1:] == messages[2:]


@pytest.mark.asyncio
async def test_only_boundary_at_index_zero_is_a_no_op():
    store = FakeSessionStore()
    sid = new_id(SessionId)
    # A single giant "turn": one USER message, then only ASSISTANT/TOOL
    # messages with no further USER boundary until the very end of the
    # window under consideration. The only safe cut is index 0.
    messages = [_user(sid, "u0")]
    for i in range(11):
        messages.append(_assistant(sid, f"a{i}"))
    await _seed(store, sid, messages)
    assert len(messages) == 12

    result = await compact_history(store, sid, _fake_summarizer, keep_recent=6, min_messages=12)

    assert result.compacted is False
    assert result.summarized_count == 0
    assert result.kept_count == 12
    assert await store.list_messages(sid) == messages


@pytest.mark.asyncio
async def test_summarizer_receives_exactly_the_prefix():
    store = FakeSessionStore()
    sid = new_id(SessionId)
    messages = []
    for i in range(6):
        messages.append(_user(sid, f"user {i}"))
        messages.append(_assistant(sid, f"assistant {i}"))
    await _seed(store, sid, messages)

    received: list[Message] = []

    async def capturing_summarizer(prefix: list[Message]) -> str:
        received.extend(prefix)
        return "ok"

    result = await compact_history(store, sid, capturing_summarizer, keep_recent=4, min_messages=12)

    assert result.compacted is True
    assert received == messages[:8]
    assert [m.id for m in received] == [m.id for m in messages[:8]]


@pytest.mark.asyncio
async def test_store_contents_after_compaction_match_result_counts():
    store = FakeSessionStore()
    sid = new_id(SessionId)
    messages = []
    for i in range(7):
        messages.append(_user(sid, f"user {i}"))
        messages.append(_assistant(sid, f"assistant {i}"))
    await _seed(store, sid, messages)  # 14 messages

    result = await compact_history(store, sid, _fake_summarizer, keep_recent=4, min_messages=12)

    new_history = await store.list_messages(sid)
    assert len(new_history) == result.kept_count + 1
    sess = await store.get(sid)
    assert sess is not None
    assert sess.message_count == len(new_history)
    assert new_history[0].role is MessageRole.USER
    assert new_history[0].metadata.annotations.get(COMPACTION_SUMMARY_ANNOTATION) is True
