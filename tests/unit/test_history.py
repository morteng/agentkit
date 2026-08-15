"""History replay: the slim shapes a reconnecting client gets back.

Two properties matter more than the rest and are asserted hardest: a receipt
never carries the tool's result, and every string that reaches the client has
been through the projection. Everything else here is shape.
"""

from datetime import UTC, datetime, timedelta

import pytest

from agentkit._content import ImageBlock, TextBlock, ThinkingBlock, ToolResultBlock, ToolUseBlock
from agentkit._ids import MessageId, OwnerId, SessionId, new_id
from agentkit._messages import (
    INJECTED_CORRECTION_ANNOTATION,
    Message,
    MessageMetadata,
    MessageRole,
)
from agentkit.history import (
    MAX_HISTORY_PAGE_LIMIT,
    MAX_SUMMARY_CHARS,
    AssistantHistoryItem,
    ToolHistoryItem,
    UserHistoryItem,
    build_history_items,
    history_frame,
    load_history_page,
)
from agentkit.store.fakes import FakeSessionStore

_SESSION = SessionId("acme:alice")
_T0 = datetime(2026, 8, 14, 19, 22, 3, tzinfo=UTC)


def _msg(role: MessageRole, content: list, *, at: int = 0, **meta) -> Message:
    return Message(
        id=new_id(MessageId),
        session_id=_SESSION,
        role=role,
        content=content,
        metadata=MessageMetadata(**meta),
        created_at=_T0 + timedelta(seconds=at),
    )


def _user(text: str, **meta) -> Message:
    return _msg(MessageRole.USER, [TextBlock(text=text)], **meta)


def test_a_plain_exchange_becomes_two_bubbles():
    items = build_history_items(
        [
            _user("find something to watch"),
            _msg(
                MessageRole.ASSISTANT,
                [TextBlock(text="Three that fit:")],
                at=8,
                model="deepseek/deepseek-v4-flash",
            ),
        ]
    )

    assert [i.kind for i in items] == ["user", "assistant"]
    assert isinstance(items[0], UserHistoryItem)
    assert items[0].text == "find something to watch"
    assert isinstance(items[1], AssistantHistoryItem)
    assert items[1].model == "deepseek/deepseek-v4-flash"
    assert items[1].at == _T0 + timedelta(seconds=8)


def test_thinking_is_dropped_and_a_thinking_only_message_makes_no_bubble():
    items = build_history_items(
        [_msg(MessageRole.ASSISTANT, [ThinkingBlock(text="the user probably means…")])]
    )
    assert items == []


def test_thinking_never_leaks_into_the_text_of_a_bubble():
    items = build_history_items(
        [
            _msg(
                MessageRole.ASSISTANT,
                [ThinkingBlock(text="internal"), TextBlock(text="visible")],
            )
        ]
    )
    assert len(items) == 1
    assert items[0].text == "visible"  # type: ignore[union-attr]


def test_injected_corrections_are_not_human_turns():
    items = build_history_items(
        [
            _user("real question"),
            _user("please call finalize", annotations={INJECTED_CORRECTION_ANNOTATION: True}),
        ]
    )
    assert [i.text for i in items] == ["real question"]  # type: ignore[union-attr]


def test_system_messages_never_appear():
    items = build_history_items([_msg(MessageRole.SYSTEM, [TextBlock(text="you are…")])])
    assert items == []


def _tool_exchange(*, is_error: bool = False, result_text: str = '["a","b"]') -> list[Message]:
    use = ToolUseBlock(id="call_0b9c", name="library_search", arguments={"query": "sci-fi"})
    return [
        _msg(MessageRole.ASSISTANT, [use], at=4),
        _msg(
            MessageRole.TOOL,
            [
                ToolResultBlock(
                    tool_use_id="call_0b9c",
                    content=[TextBlock(text=result_text)],
                    is_error=is_error,
                )
            ],
            at=5,
        ),
    ]


def test_a_tool_call_becomes_a_receipt_with_no_result_content():
    """The non-negotiable one: a reload must not replay what a tool returned."""
    items = build_history_items(_tool_exchange(result_text='["a","b","c"]'))

    assert len(items) == 1
    receipt = items[0]
    assert isinstance(receipt, ToolHistoryItem)
    assert receipt.id == "call_0b9c"
    assert receipt.tool_name == "library_search"
    assert receipt.status == "ok"
    assert receipt.summary == "3 results"
    assert receipt.arg_preview == {"query": "sci-fi"}
    # There is no field to put the payload in, and none appears in the dump.
    dumped = receipt.model_dump(mode="json")
    assert "content" not in dumped
    assert "a" not in dumped.values()


def test_an_unanswered_call_is_pending_not_missing():
    use = ToolUseBlock(id="call_x", name="torrent_add", arguments={})
    items = build_history_items([_msg(MessageRole.ASSISTANT, [use])])
    assert items[0].status == "pending"  # type: ignore[union-attr]


def test_a_failed_call_summarises_the_error():
    items = build_history_items(_tool_exchange(is_error=True, result_text="upstream 500"))
    assert items[0].status == "error"  # type: ignore[union-attr]
    assert items[0].summary == "upstream 500"  # type: ignore[union-attr]


def test_a_media_result_is_counted_never_embedded():
    use = ToolUseBlock(id="call_i", name="library_art", arguments={})
    result = ToolResultBlock(
        tool_use_id="call_i",
        content=[ImageBlock(media_type="image/png", data="AAAABBBB")],
    )
    items = build_history_items(
        [
            _msg(MessageRole.ASSISTANT, [use]),
            _msg(MessageRole.TOOL, [result]),
        ]
    )
    assert items[0].summary == "1 image"  # type: ignore[union-attr]
    assert "AAAABBBB" not in str(items[0].model_dump(mode="json"))


def test_a_wrapped_json_payload_still_counts_its_items():
    items = build_history_items(_tool_exchange(result_text='{"Items": [1, 2, 3, 4]}'))
    assert items[0].summary == "4 results"  # type: ignore[union-attr]


def test_prose_falls_back_to_a_character_count():
    items = build_history_items(_tool_exchange(result_text="hello"))
    assert items[0].summary == "5 chars"  # type: ignore[union-attr]


def test_a_summary_is_capped_and_stripped_of_control_characters():
    """An error string is attacker-influenced text on its way to a UI."""
    # A NUL (Cc), a bidi override (Cf) that makes a rendered string lie about
    # its content, a line separator (Zl) that breaks a chat client's layout,
    # and enough length to need the cap.
    hostile = "boom\x00 \u202ereversed\u2028second line " + "x" * 400
    items = build_history_items(_tool_exchange(is_error=True, result_text=hostile))
    summary = items[0].summary  # type: ignore[union-attr]

    assert len(summary) <= MAX_SUMMARY_CHARS
    assert "\x00" not in summary
    assert "\u202e" not in summary
    assert "\u2028" not in summary
    # Whitespace runs are collapsed, so nothing can be hidden in a gap.
    assert "  " not in summary


def test_argument_previews_mask_secrets_and_flatten_structure():
    use = ToolUseBlock(
        id="call_a",
        name="provision",
        arguments={
            "user": "bob",
            "password": "hunter2",
            "opts": {"a": 1},
            "tags": [1, 2],
            "note": "n" * 300,
        },
    )
    items = build_history_items([_msg(MessageRole.ASSISTANT, [use])])
    preview = items[0].arg_preview  # type: ignore[union-attr]

    assert preview["password"] == "***"
    assert preview["opts"] == "[object]"
    assert preview["tags"] == "[array]"
    assert len(preview["note"]) <= 120
    assert items[0].args_truncated is False  # type: ignore[union-attr]


def test_too_many_arguments_are_cut_and_flagged():
    use = ToolUseBlock(id="call_a", name="wide", arguments={f"k{i:02d}": i for i in range(20)})
    items = build_history_items([_msg(MessageRole.ASSISTANT, [use])])
    assert len(items[0].arg_preview) == 8  # type: ignore[union-attr]
    assert items[0].args_truncated is True  # type: ignore[union-attr]


def test_a_message_with_text_and_calls_emits_the_bubble_then_the_receipts():
    items = build_history_items(
        [
            _msg(
                MessageRole.ASSISTANT,
                [
                    TextBlock(text="looking that up"),
                    ToolUseBlock(id="c1", name="a", arguments={}),
                    ToolUseBlock(id="c2", name="b", arguments={}),
                ],
            )
        ]
    )
    assert [i.kind for i in items] == ["assistant", "tool", "tool"]
    assert [i.id for i in items[1:]] == ["c1", "c2"]


@pytest.mark.asyncio
async def test_an_empty_session_is_an_empty_page_not_an_error():
    store = FakeSessionStore()
    page = await load_history_page(store, _SESSION)
    assert page.count == 0
    assert page.items == []
    assert page.truncated is False
    assert page.session_id == "acme:alice"


@pytest.mark.asyncio
async def test_an_oversized_limit_is_clamped_not_rejected():
    store = FakeSessionStore()
    await store.create(_SESSION, OwnerId("acme:alice"))
    for i in range(5):
        await store.append_message(_SESSION, _user(f"m{i}"))

    page = await load_history_page(store, _SESSION, limit=10_000)
    assert page.count == 5
    assert page.truncated is False

    # And the floor: 0 or a negative page size still returns something.
    assert (await load_history_page(store, _SESSION, limit=0)).count == 1


@pytest.mark.asyncio
async def test_a_full_page_says_there_is_more_above_it():
    store = FakeSessionStore()
    await store.create(_SESSION, OwnerId("acme:alice"))
    for i in range(6):
        await store.append_message(_SESSION, _user(f"m{i}"))

    page = await load_history_page(store, _SESSION, limit=3)
    assert page.truncated is True
    # Oldest-first within the window, and it is the newest window.
    assert [i.text for i in page.items] == ["m3", "m4", "m5"]  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_count_counts_items_not_store_messages():
    """One message can expand to a bubble plus several receipts."""
    store = FakeSessionStore()
    await store.create(_SESSION, OwnerId("acme:alice"))
    await store.append_message(
        _SESSION,
        _msg(
            MessageRole.ASSISTANT,
            [
                TextBlock(text="hi"),
                ToolUseBlock(id="c1", name="a", arguments={}),
            ],
        ),
    )
    page = await load_history_page(store, _SESSION, limit=50)
    assert page.count == 2


@pytest.mark.asyncio
async def test_the_ws_frame_is_the_rest_body_plus_a_type():
    store = FakeSessionStore()
    page = await load_history_page(store, _SESSION)
    frame = history_frame(page)

    assert frame["type"] == "history"
    assert set(frame) == {"type", "session_id", "count", "truncated", "items"}
    # Not a BaseEvent: no turn owns it.
    assert "event_id" not in frame
    assert "sequence" not in frame


def test_the_page_ceiling_is_the_documented_one():
    assert MAX_HISTORY_PAGE_LIMIT == 200
