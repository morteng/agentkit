"""Replay-safe projections of a stored transcript.

Every consumer of a persisted :class:`~agentkit.store.session.SessionStore`
needs the same thing when a client reconnects: what was said, and the fact that
a tool ran. Left to each consumer, that gets hand-rolled N times into N subtly
different shapes — and, worse, N different answers to the one question that
actually matters here, which is *how much of a tool result goes back on the
page*.

The safe answer is none of it, and it is not the obvious one. Rehydrating
result payloads replays stale media grids on every reload and re-injects
third-party text into the page outside any turn, where none of the turn's
provenance handling applies. So a receipt built here says **that** a tool ran —
name, status, a server-built summary, a flattened argument preview — and never
what it returned. There is no field for the content, so there is nothing for a
consumer to accidentally render.

The shapes match section 2 of REDACTED's ``API-CONTRACT.md`` exactly, so a
consumer can serve :func:`load_history_page` from a REST route, replay it over
the WebSocket transport (``replay_history=True``, see
:func:`agentkit.transports.websocket.mount_websocket_route`), or both, and the
client parses one shape either way.
"""

from __future__ import annotations

import json
from datetime import datetime  # noqa: TC003 — pydantic resolves this at runtime
from typing import TYPE_CHECKING, Annotated, Any, Literal, cast

from pydantic import BaseModel, Field

from agentkit._content import ImageBlock, TextBlock, ToolResultBlock, ToolUseBlock
from agentkit._messages import INJECTED_CORRECTION_ANNOTATION, Message, MessageRole
from agentkit._redaction import argument_preview, clean_text

if TYPE_CHECKING:
    from collections.abc import Sequence

    from agentkit._ids import SessionId
    from agentkit.store.session import SessionStore

#: Default number of stored messages a replay loads.
DEFAULT_HISTORY_PAGE_LIMIT = 50

#: Hard ceiling on that number. Limits are clamped into ``1..MAX`` rather than
#: rejected: an out-of-range page size is a client bug that must not cost the
#: user their transcript.
MAX_HISTORY_PAGE_LIMIT = 200

#: Cap on a receipt summary. It is display text, not data.
MAX_SUMMARY_CHARS = 200

#: WebSocket frame type for a replayed page. Not a
#: :class:`~agentkit.events.base.BaseEvent`: it has no ``event_id``,
#: ``turn_id`` or ``sequence``, because it belongs to no turn.
HISTORY_FRAME_TYPE = "history"

_LIST_WRAPPER_KEYS = ("items", "results", "data", "entries")
"""Keys whose list value counts as "the results" in a wrapped JSON payload —
``{"Items": [...]}`` and friends. Matched case-insensitively."""


class UserHistoryItem(BaseModel):
    """A human turn. ``text`` is plain text — never rendered as markdown."""

    id: str
    kind: Literal["user"] = "user"
    at: datetime
    text: str


class AssistantHistoryItem(BaseModel):
    """An assistant reply.

    Thinking content is dropped entirely rather than collapsed or truncated: a
    reasoning trace is not part of the conversation, and putting one back on
    the page hours later shows the user something the live stream deliberately
    kept separate.
    """

    id: str
    kind: Literal["assistant"] = "assistant"
    at: datetime
    text: str
    model: str | None = None


class ToolHistoryItem(BaseModel):
    """A tool-call receipt: that it ran, not what it returned."""

    id: str
    kind: Literal["tool"] = "tool"
    at: datetime
    tool_name: str
    status: Literal["ok", "error", "pending"]
    """``"pending"`` when no result for this call is in the loaded window — the
    turn was cut off, or the result is older than the limit. Either way the
    receipt is honest about not knowing."""
    summary: str
    arg_preview: dict[str, Any]
    args_truncated: bool


HistoryItem = Annotated[
    UserHistoryItem | AssistantHistoryItem | ToolHistoryItem,
    Field(discriminator="kind"),
]


class HistoryPage(BaseModel):
    """One page of replayed transcript, oldest-first."""

    session_id: str
    count: int
    """Number of ``items`` — not of store messages. One message can expand to
    an assistant bubble plus several receipts."""
    truncated: bool
    """The store had at least as many messages as the limit allowed, so older
    history exists above the first item."""
    items: list[HistoryItem]


def _joined_text(message: Message) -> str:
    return "\n".join(b.text for b in message.content if isinstance(b, TextBlock))


def _result_index(messages: Sequence[Message]) -> dict[str, ToolResultBlock]:
    """Map ``tool_use_id`` to the result block answering it, first one wins."""
    index: dict[str, ToolResultBlock] = {}
    for message in messages:
        for block in message.content:
            if isinstance(block, ToolResultBlock) and block.tool_use_id not in index:
                index[block.tool_use_id] = block
    return index


def _json_item_count(text: str) -> int | None:
    """Number of top-level items when ``text`` is a JSON list or a list wrapper."""
    stripped = text.strip()
    if not stripped or stripped[0] not in "[{":
        return None
    try:
        parsed: Any = json.loads(stripped)
    except ValueError:
        return None
    if isinstance(parsed, list):
        return len(cast("list[Any]", parsed))
    if isinstance(parsed, dict):
        mapping = cast("dict[Any, Any]", parsed)
        lowered = {k.lower(): v for k, v in mapping.items() if isinstance(k, str)}
        for key in _LIST_WRAPPER_KEYS:
            value = lowered.get(key)
            if isinstance(value, list):
                return len(cast("list[Any]", value))
    return None


def summarize_result(result: ToolResultBlock | None) -> str:
    """Build the one-line summary a receipt carries.

    Server-built and capped, in this order: the item count when the result
    parses as a list (or a ``{"Items": [...]}`` wrapper), then the error text
    when the call failed, then an image count when the result is only media,
    then a character count. Anything derived from the result is passed through
    :func:`~agentkit._redaction.clean_text` first — the content is
    third-party text and this string ends up in a UI.
    """
    if result is None:
        return ""
    text = "\n".join(b.text for b in result.content if isinstance(b, TextBlock))
    images = sum(1 for b in result.content if isinstance(b, ImageBlock))

    count = _json_item_count(text)
    if count is not None:
        summary = f"{count} results"
    elif result.is_error:
        summary = text or "error"
    elif not text and images:
        summary = f"{images} image{'s' if images != 1 else ''}"
    else:
        summary = f"{len(text)} chars"
    return clean_text(summary, limit=MAX_SUMMARY_CHARS)


def _tool_item(
    block: ToolUseBlock, message: Message, result: ToolResultBlock | None
) -> ToolHistoryItem:
    preview, truncated = argument_preview(block.arguments)
    if result is None:
        status: Literal["ok", "error", "pending"] = "pending"
    elif result.is_error:
        status = "error"
    else:
        status = "ok"
    return ToolHistoryItem(
        id=block.id,
        at=message.created_at,
        tool_name=block.name,
        status=status,
        summary=summarize_result(result),
        arg_preview=preview,
        args_truncated=truncated,
    )


def build_history_items(messages: Sequence[Message]) -> list[HistoryItem]:
    """Project stored messages to client items, oldest-first.

    Skipped: SYSTEM messages; USER messages the loop injected as corrections
    (they are re-prompts, not human turns); assistant messages with no text
    (thinking- or tool-only, which contribute receipts and no bubble). An
    assistant message that has both text and tool calls emits its bubble first,
    then one receipt per call in block order.

    A TOOL message is never emitted on its own — it is only read for the
    results that complete a receipt. That includes the synthetic cancelled
    results agentkit writes to close an abandoned call
    (:data:`agentkit.session.SYNTHETIC_TOOL_RESULT_ANNOTATION`), which
    correctly surface as an ``error`` receipt: the call really did not produce
    a result.
    """
    results = _result_index(messages)
    items: list[HistoryItem] = []
    for message in messages:
        if message.role is MessageRole.SYSTEM or message.role is MessageRole.TOOL:
            continue
        text = _joined_text(message)
        if message.role is MessageRole.USER:
            if INJECTED_CORRECTION_ANNOTATION in message.metadata.annotations:
                continue
            if text:
                items.append(UserHistoryItem(id=str(message.id), at=message.created_at, text=text))
            continue
        if text:
            items.append(
                AssistantHistoryItem(
                    id=str(message.id),
                    at=message.created_at,
                    text=text,
                    model=message.metadata.model,
                )
            )
        for block in message.content:
            if isinstance(block, ToolUseBlock):
                items.append(_tool_item(block, message, results.get(block.id)))
    return items


async def load_history_page(
    store: SessionStore,
    session_id: SessionId,
    *,
    limit: int = DEFAULT_HISTORY_PAGE_LIMIT,
    max_limit: int = MAX_HISTORY_PAGE_LIMIT,
) -> HistoryPage:
    """Load and project the newest ``limit`` stored messages for ``session_id``.

    ``limit`` is clamped into ``1..max_limit`` rather than validated: a client
    asking for 10_000 messages gets 200, not an error page where its history
    should be. An empty or unknown session is a normal empty page — that is the
    signal a UI renders its first-run state from, not a failure.
    """
    bounded = max(1, min(limit, max_limit))
    messages = await store.list_messages(session_id, limit=bounded)
    items = build_history_items(messages)
    return HistoryPage(
        session_id=str(session_id),
        count=len(items),
        # Exactly ``bounded`` messages back means the store had at least that
        # many; it may have had more. Counted in store messages, not items.
        truncated=len(messages) >= bounded,
        items=items,
    )


def history_frame(page: HistoryPage) -> dict[str, Any]:
    """The WebSocket frame carrying ``page``: the REST body plus a ``type``."""
    return {"type": HISTORY_FRAME_TYPE, **page.model_dump(mode="json")}
