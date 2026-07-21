"""Conversation-history compaction.

Hosts own model selection and billing — agentkit never calls an LLM itself.
:func:`compact_history` takes a host-supplied ``summarizer`` callback, uses it
to condense the older half of a session's transcript into a single message,
and writes the result back through :meth:`SessionStore.replace`.

This module ships the primitive only. There is no automatic trigger wired
into the loop: a host invokes ``compact_history`` from its own seam — e.g.
its ``AgentConfig.on_iteration_start`` hook (see ``session.py``/``config.py``),
a scheduled job, or an explicit "compact this session" action. Auto-compaction
policy (when to trigger, how aggressively) is host-specific and ships
separately, if at all.

The one invariant agentkit *does* enforce here: the cut point never splits a
``ToolUseBlock``/``ToolResultBlock`` pair across the summary/kept-tail
boundary. Providers such as Gemini and Anthropic hard-reject a request whose
history contains an orphaned tool result (a ``tool_result`` with no
preceding matching ``tool_use``) — this is the entire reason boundary
selection lives in agentkit rather than being left to each host to
reimplement.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from agentkit._content import TextBlock, ToolResultBlock, ToolUseBlock
from agentkit._ids import MessageId, new_id
from agentkit._messages import INJECTED_CORRECTION_ANNOTATION, Message, MessageMetadata, MessageRole

if TYPE_CHECKING:
    from agentkit._ids import SessionId
    from agentkit.store.session import SessionStore

#: ``Message.metadata.annotations`` key marking a USER message that
#: ``compact_history`` synthesized to replace an earlier span of the
#: transcript. Mirrors ``INJECTED_CORRECTION_ANNOTATION`` — lets a consumer
#: (or another compaction pass) distinguish a real human prompt from a
#: condensed summary standing in for one.
COMPACTION_SUMMARY_ANNOTATION = "compaction_summary"

#: Prefix prepended to every summary message so it is unambiguous in
#: transcripts/UIs — bilingual (nb/en) per project convention for
#: user-visible strings that might surface in either editor language.
_SUMMARY_PREFIX = "[Samtalesammendrag / conversation summary — earlier messages were condensed]\n\n"

Summarizer = Callable[[list[Message]], Awaitable[str]]


@dataclass(frozen=True)
class CompactionResult:
    """Outcome of a :func:`compact_history` call."""

    summarized_count: int
    kept_count: int
    summary: str
    compacted: bool


def _tool_pair_spans(history: list[Message]) -> list[tuple[int, int]]:
    """Return ``(use_idx, result_idx)`` for every tool_use/tool_result pair.

    ``use_idx``/``result_idx`` are indices into ``history`` of the messages
    carrying the ``ToolUseBlock`` and its matching ``ToolResultBlock``
    respectively. A tool_use with no result yet (still in flight, or the
    transcript was truncated) simply produces no span — nothing to protect.
    """
    use_idx_by_id: dict[str, int] = {}
    for idx, msg in enumerate(history):
        for block in msg.content:
            if isinstance(block, ToolUseBlock):
                use_idx_by_id[block.id] = idx

    spans: list[tuple[int, int]] = []
    for idx, msg in enumerate(history):
        for block in msg.content:
            if isinstance(block, ToolResultBlock):
                use_idx = use_idx_by_id.get(block.tool_use_id)
                if use_idx is not None:
                    spans.append((use_idx, idx))
    return spans


def _is_safe_boundary(history: list[Message], i: int, spans: list[tuple[int, int]]) -> bool:
    """Is index ``i`` safe to cut at — i.e. can become the first kept message?

    Safe means: ``history[i]`` is a genuine USER turn boundary (not a
    loop-injected correction, not a pure tool-result carrier — the latter
    doesn't occur in this codebase's canonical Message shape today, since
    tool results are their own ``MessageRole.TOOL`` messages, but the check
    costs nothing and guards against a future/foreign transcript shape), and
    no tool_use/tool_result span straddles the cut (use before ``i``, result
    at-or-after ``i``).
    """
    msg = history[i]
    if msg.role is not MessageRole.USER:
        return False
    if msg.metadata.annotations.get(INJECTED_CORRECTION_ANNOTATION):
        return False
    if msg.content and all(isinstance(b, ToolResultBlock) for b in msg.content):
        return False
    return not any(use_idx < i <= result_idx for use_idx, result_idx in spans)


def _find_cut_index(history: list[Message], keep_recent: int) -> int | None:
    """Find the nearest safe cut at or before ``len(history) - keep_recent``.

    Walks backward from the naive cut point toward (but not including) index
    0 — a cut at 0 would summarize nothing, defeating the point — and
    returns the first index that satisfies :func:`_is_safe_boundary`.
    Returns ``None`` when no such index exists (index 0 is the only safe
    boundary, or the naive cut point is already at/before 0).
    """
    naive_cut = len(history) - keep_recent
    if naive_cut <= 0:
        return None
    spans = _tool_pair_spans(history)
    for i in range(naive_cut, 0, -1):
        if _is_safe_boundary(history, i, spans):
            return i
    return None


async def compact_history(
    store: SessionStore,
    session_id: SessionId,
    summarizer: Summarizer,
    *,
    keep_recent: int = 8,
    min_messages: int = 12,
) -> CompactionResult:
    """Condense the older portion of a session's transcript into one message.

    Loads the full transcript via ``store.list_messages``. If it has fewer
    than ``min_messages`` messages, this is a no-op — returns
    ``compacted=False`` and leaves the store untouched. Otherwise, the
    trailing ``keep_recent`` messages are kept verbatim, with the cut point
    moved earlier (never later) to the nearest boundary where the kept tail
    starts at a genuine user turn and no tool_use/tool_result pair is split
    across it. If no such boundary exists short of index 0, this is also a
    no-op (``compacted=False``) — there's nothing safe to summarize.

    Otherwise, ``summarizer`` is awaited with exactly the messages being
    dropped (the prefix before the cut) and must return the summary text.
    agentkit does not call an LLM itself; the host owns that call — model
    choice, prompting, cost. The new transcript
    ``[summary_message, *kept_tail]`` is written back via
    ``store.replace`` in one atomic step.

    Raises whatever ``store.replace``/``store.list_messages`` raise (e.g. a
    missing session) and propagates any exception from ``summarizer``
    unmodified — a failed summarization leaves the store untouched, since
    ``replace`` is only called after ``summarizer`` returns successfully.
    """
    # SessionStore.list_messages defaults to the last 200 messages; compaction
    # needs the FULL transcript to find a safe cut, so pass an explicit
    # high-water limit rather than relying on the default.
    history = await store.list_messages(session_id, limit=1_000_000)

    if len(history) < min_messages:
        return CompactionResult(
            summarized_count=0, kept_count=len(history), summary="", compacted=False
        )

    cut = _find_cut_index(history, keep_recent)
    if cut is None:
        return CompactionResult(
            summarized_count=0, kept_count=len(history), summary="", compacted=False
        )

    prefix = history[:cut]
    tail = history[cut:]

    summary_text = await summarizer(prefix)

    summary_message = Message(
        id=new_id(MessageId),
        session_id=session_id,
        role=MessageRole.USER,
        content=[TextBlock(text=f"{_SUMMARY_PREFIX}{summary_text}")],
        metadata=MessageMetadata(annotations={COMPACTION_SUMMARY_ANNOTATION: True}),
        created_at=datetime.now(UTC),
    )

    new_history = [summary_message, *tail]
    await store.replace(session_id, new_history)

    return CompactionResult(
        summarized_count=len(prefix),
        kept_count=len(tail),
        summary=summary_text,
        compacted=True,
    )
