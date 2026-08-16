"""AgentSession — top-level entry point for consumers.

Wraps a single conversation: persistent SessionStore-backed history, plus a
``run(user_message)`` async-iterator API that yields events. Internally
constructs a Loop per turn.

Turn durability is designed once, here, and shared by every entry point
(``run``, ``resume_with_approval``, ``resume_with_approval_batch``):

* :class:`_TurnRunner` owns the background Loop task for one turn and runs
  end-of-turn cleanup — cancel the task, then persist what the turn produced —
  exactly once, triggered by whichever comes first: the stream reaching
  ``TurnEnded``, or the ``async with`` block exiting. Cleanup is therefore tied
  to the context manager, not to garbage collection of a half-drained
  generator.
* Persistence never writes a malformed transcript. A turn that ends without a
  result for some ``ToolUseBlock`` (cancelled, errored, or simply abandoned)
  gets a synthesized ``cancelled`` ``ToolResultBlock`` for each unanswered
  call, because both the Anthropic and OpenAI protocols reject a history whose
  ``tool_use`` has no matching ``tool_result`` — one such turn would otherwise
  break every subsequent turn in the session. The one exception is a turn that
  suspended on approval: its dangling calls are owned by the checkpoint and
  resolved on resume.
* History loading is bounded by an explicit ``history_limit`` and the cut is
  moved forward to a tool-pair-safe boundary, with a warning when messages are
  dropped.
"""

from __future__ import annotations

import asyncio
import contextlib
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from agentkit._content import TextBlock, ToolResultBlock, ToolUseBlock
from agentkit._ids import CheckpointId, EventId, MessageId, OwnerId, SessionId, TurnId, new_id
from agentkit._logging import get_logger
from agentkit._messages import Message, MessageMetadata, MessageRole
from agentkit._redaction import argument_preview, clean_text

# ``_tool_pair_spans`` is imported rather than reimplemented: pairing tool_use
# to tool_result must behave identically in compaction's cut-point search and
# in this module's truncation cut, or the two would disagree about what a safe
# boundary is. It is private to the package, not to the module.
from agentkit.audit import (
    ACTION_APPROVAL_RESOLVED,
    SOURCE_AGENT,
    SOURCE_HUMAN,
    AuditRecord,
    AuditSink,
    NullAuditSink,
    record_audit,
)
from agentkit.compaction import (
    _tool_pair_spans,  # pyright: ignore[reportPrivateUsage]
)
from agentkit.errors import ApprovalTimeout, CheckpointMissing
from agentkit.events import (
    ApprovalDenied,
    ApprovalGranted,
    ApprovalResolved,
    ErrorCode,
    Errored,
    Event,
    TurnEnded,
    TurnEndReason,
    TurnMetrics,
)
from agentkit.events.approval import SYSTEM_RESOLVER
from agentkit.guards.approval import RiskBasedApprovalGate
from agentkit.guards.finalize import StructuralFinalizeValidator
from agentkit.guards.taint import TaintSource
from agentkit.loop.context import SystemClock, TurnContext, from_checkpoint_payload
from agentkit.loop.handlers.approval_wait import handle_approval_wait
from agentkit.loop.handlers.context_build import handle_context_build
from agentkit.loop.handlers.finalize_check import handle_finalize_check
from agentkit.loop.handlers.intent_gate import handle_intent_gate
from agentkit.loop.handlers.memory_extract import handle_memory_extract
from agentkit.loop.handlers.streaming import handle_streaming
from agentkit.loop.handlers.tool_executing import handle_tool_executing
from agentkit.loop.handlers.tool_phase import handle_tool_phase
from agentkit.loop.handlers.tool_results import handle_tool_results
from agentkit.loop.message_builder import MessageBuilder
from agentkit.loop.orchestrator import Loop
from agentkit.loop.phase import Phase
from agentkit.loop.tool_dispatcher import DispatchPolicy, ToolDispatcher
from agentkit.subagents.dispatcher import SubagentDispatcher

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator, Callable, Coroutine, Sequence

    from agentkit.config import AgentConfig
    from agentkit.providers.base import Provider, SystemBlock
    from agentkit.store.session import SessionStore
    from agentkit.tools.registry import ToolRegistry
    from agentkit.tools.spec import ToolSpec

log = get_logger(__name__)

#: How many stored messages ``run()`` loads into a new turn by default. Equal
#: to ``SessionStore.list_messages``'s own default, so the window is unchanged
#: for existing consumers — the difference is that it is now explicit, the cut
#: is tool-pair-safe, and dropping messages is logged instead of silent. Pass
#: ``history_limit=`` to :class:`AgentSession` to widen it.
DEFAULT_HISTORY_LIMIT = 200

#: ``Message.metadata.annotations`` key marking a TOOL message that agentkit
#: synthesized to close a ``ToolUseBlock`` which never received a real result.
#: Mirrors ``INJECTED_CORRECTION_ANNOTATION``/``COMPACTION_SUMMARY_ANNOTATION``:
#: a consumer rendering a transcript can tell a real tool result from a
#: placeholder agentkit wrote to keep the history well-formed. The value is the
#: synthetic result's status, currently always ``"cancelled"``.
SYNTHETIC_TOOL_RESULT_ANNOTATION = "synthetic_tool_result"

_CANCELLED_STATUS = "cancelled"

_CANCELLED_TOOL_RESULT_TEXT = "cancelled: the turn ended before this tool call produced a result"

#: Deny reason recorded for approvals a single-call resume did not rule on.
_UNRESOLVED_DENY_REASON = "not resolved in this resume"

#: Cap on the deny reason copied into an audit record. It is client-supplied.
_MAX_AUDIT_REASON_CHARS = 200


def _unresolved_tool_use_ids(history: Sequence[Message]) -> list[str]:
    """Return the ``tool_use`` ids in ``history`` that no tool_result answers.

    Same two-pass id-map shape as :func:`agentkit.compaction._tool_pair_spans`,
    but resolved at *block* granularity rather than to message indices: one
    assistant message can carry several parallel ``ToolUseBlock``s of which
    only some were answered, and the unanswered ones are exactly what has to be
    closed off. Ids come back in the order the calls were made.
    """
    unresolved: dict[str, None] = {}
    for msg in history:
        for block in msg.content:
            if isinstance(block, ToolUseBlock):
                unresolved[block.id] = None
    for msg in history:
        for block in msg.content:
            if isinstance(block, ToolResultBlock):
                unresolved.pop(block.tool_use_id, None)
    return list(unresolved)


def _cancelled_tool_result(
    session_id: SessionId, tool_use_id: str, *, created_at: datetime
) -> Message:
    """A TOOL message closing ``tool_use_id`` with a ``cancelled`` result."""
    return Message(
        id=new_id(MessageId),
        session_id=session_id,
        role=MessageRole.TOOL,
        content=[
            ToolResultBlock(
                tool_use_id=tool_use_id,
                content=[TextBlock(text=_CANCELLED_TOOL_RESULT_TEXT)],
                is_error=True,
            )
        ],
        metadata=MessageMetadata(annotations={SYNTHETIC_TOOL_RESULT_ANNOTATION: _CANCELLED_STATUS}),
        created_at=created_at,
    )


def _repair_dangling_tool_uses(
    history: list[Message], session_id: SessionId
) -> tuple[list[Message], int]:
    """Insert a cancelled result after every message with an unanswered tool_use.

    Returns ``(history, repaired_count)`` — the input list itself when there
    was nothing to repair. Position matters: Anthropic requires the results for
    a ``tool_use`` in the message that immediately follows it, so a repair has
    to be spliced in place rather than appended at the end.

    This is an in-memory repair of what agentkit is about to *send*; the stored
    transcript is left alone. A store rewrite would need
    :meth:`SessionStore.replace` and would race the very turns that write to
    the same session — and with cancelled turns now persisting their own
    closing results, the only transcripts needing this are legacy ones and
    approval-suspended turns that were never resumed (whose dangling calls are
    still legitimately owned by a live checkpoint).
    """
    unresolved = set(_unresolved_tool_use_ids(history))
    if not unresolved:
        return history, 0
    repaired: list[Message] = []
    for msg in history:
        repaired.append(msg)
        for block in msg.content:
            if isinstance(block, ToolUseBlock) and block.id in unresolved:
                repaired.append(
                    _cancelled_tool_result(session_id, block.id, created_at=msg.created_at)
                )
    return repaired, len(unresolved)


def _tool_safe_start(messages: Sequence[Message], start: int) -> int:
    """Move ``start`` forward until ``messages[start:]`` orphans no tool_result.

    A truncating load keeps a suffix of the transcript, and a naive suffix can
    begin *after* a ``tool_use`` but *before* its ``tool_result`` — which every
    provider rejects. Pair resolution is delegated to
    :func:`agentkit.compaction._tool_pair_spans` so the two cut-point searches
    in this package cannot drift apart. A tool_result whose ``tool_use`` fell
    outside the fetched window produces no span at all, and is treated as
    orphaned for the same reason.
    """
    if start <= 0:
        return 0
    uses_by_result: dict[int, list[int]] = {}
    for use_idx, result_idx in _tool_pair_spans(list(messages)):
        uses_by_result.setdefault(result_idx, []).append(use_idx)

    cut = start
    for idx in range(start, len(messages)):
        result_blocks = sum(1 for b in messages[idx].content if isinstance(b, ToolResultBlock))
        if not result_blocks:
            continue
        matched = uses_by_result.get(idx, [])
        if len(matched) < result_blocks or min(matched) < cut:
            # This message's results would be orphaned by a cut at ``cut``;
            # the earliest safe boundary is past it. ``cut`` only ever grows,
            # and every index it passes is excluded from the kept tail, so one
            # forward pass settles it.
            cut = idx + 1
    return cut


class _TurnRunner:
    """Drives one turn's Loop in a background task and owns its cleanup.

    Cleanup — cancelling the task, then persisting the messages the turn added
    to ``ctx.history`` — runs exactly once, whichever happens first:

    * the consumer drains :meth:`stream` to ``TurnEnded`` (the generator's
      ``finally`` fires on the way out), or
    * the ``async with`` block exits, drained or not.

    The second trigger is the point: exiting the context manager without
    draining used to leave the Loop task running detached, with cancellation
    and persistence deferred to garbage collection of the abandoned generator.

    ``drain`` and ``persist`` are passed in as callables rather than the
    session itself: the runner is a lifecycle primitive and has no business
    knowing what a session is.
    """

    def __init__(
        self,
        *,
        queue: asyncio.Queue[Any],
        drain: Callable[[], Coroutine[Any, Any, None]],
        persist: Callable[[TurnEndReason | None], Coroutine[Any, Any, None]],
    ) -> None:
        self._queue = queue
        self._drain = drain
        self._persist = persist
        self._task: asyncio.Task[None] | None = None
        self._end_reason: TurnEndReason | None = None
        self._finished = False

    def stream(self) -> AsyncIterator[Event]:
        """The turn's event stream. Starts the Loop on first iteration."""
        return self._iter()

    async def _iter(self) -> AsyncIterator[Event]:
        if self._task is None:
            self._task = asyncio.create_task(self._drain())
        try:
            while True:
                item = await self._queue.get()
                yield item
                if isinstance(item, TurnEnded):
                    self._end_reason = item.reason
                    break
        finally:
            await self.finish()

    async def finish(self) -> None:
        """Cancel the Loop task and persist the turn. Idempotent."""
        if self._finished:
            return
        self._finished = True
        task = self._task
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        # Shielded: when the *consumer's* task is the thing being cancelled,
        # an unshielded await here would raise before the writes land — losing
        # precisely the turn whose record matters most.
        await asyncio.shield(self._persist(self._end_reason))


class _SessionToolAuthorizer:
    """The execution-time gate :class:`AgentSession` installs on its registry.

    Holds a reference to ``base`` — whatever the consumer had installed before
    the session touched the registry — so a second session on the same registry
    rebuilds the chain from that base rather than wrapping the first session's
    chain. Without the marker, N sessions on a shared registry would nest N
    chains deep and re-run the consumer's gate N times per call.
    """

    def __init__(self, *, base: Any, chain: Any) -> None:
        self.base = base
        self._chain = chain

    def authorize(self, spec: ToolSpec, ctx: Any) -> str | None:
        result: str | None = self._chain.authorize(spec, ctx)
        return result


class AgentSession:
    """One conversation. Run turns via ``async with session.run(text) as stream:``."""

    def __init__(
        self,
        *,
        owner: OwnerId,
        config: AgentConfig,
        registry: ToolRegistry,
        model: str,
        provider: Provider | None = None,
        system_blocks: list[SystemBlock] | None = None,
        session_id: SessionId | None = None,
        history_limit: int = DEFAULT_HISTORY_LIMIT,
        audit_sink: AuditSink | None = None,
    ) -> None:
        selector = config.provider_selector
        if provider is None and selector is None:
            raise ValueError(
                "AgentSession needs either a provider or a config.provider_selector — got neither."
            )
        if provider is not None and selector is not None:
            raise ValueError(
                "AgentSession got both a provider and a config.provider_selector — set exactly one."
            )
        self.owner = owner
        self.id: SessionId = session_id or new_id(SessionId)
        self.config = config
        self.provider = provider
        self.registry = registry
        self.model = model
        self.system_blocks = system_blocks or []
        #: Upper bound on how many stored messages a new turn loads. Raising it
        #: costs context/tokens; lowering it drops older turns sooner. Either
        #: way the cut is tool-pair-safe and logged — see :meth:`_load_history`.
        self.history_limit = history_limit
        #: Where this session's audit records go. One sink for the whole
        #: session, resolved once here, so no tool client and no call site can
        #: be the one that was never given one. ``NullAuditSink`` when the
        #: consumer configured nothing — silent, but uniformly silent. Wrap
        #: several destinations in ``MultiAuditSink``.
        self.audit_sink: AuditSink = audit_sink or config.audit_sink or NullAuditSink()
        self._initialized = False
        self._install_tool_authorizer()

    def _install_tool_authorizer(self) -> None:
        """Wire the registry's execution-time gate from this session's config.

        Filtering the advertised catalog is advisory: the model can name a tool
        it was never shown. The registry gate is what actually refuses it, so a
        session that configures a :class:`~agentkit.toolplane.ToolPlane`
        installs the matching :class:`ToolPlaneAuthorizer` rather than leaving
        the tier decision as advertising-only. ``SubagentToolAuthorizer`` is
        always chained in: it is inert outside a subagent context (it keys off
        ``ctx.metadata["allowed_tools"]``, which only the dispatcher stamps),
        and inside one it is the registry-level twin of the restricted view.

        Anything the consumer already installed is preserved and runs first —
        this composes with a hand-wired gate instead of replacing it. On a
        registry shared by several sessions the newest session's chain replaces
        the previous one (it does not nest inside it), so the consumer's own
        gate is never wrapped twice and the chain cannot grow without bound.
        """
        # Local imports: both modules import tool/loop types, and session.py is
        # imported early. Same rationale as the other deferred imports here.
        from agentkit.subagents.isolation import (  # noqa: PLC0415
            SubagentToolAuthorizer,
            chain_authorizers,
        )
        from agentkit.toolplane import ToolPlane  # noqa: PLC0415
        from agentkit.toolplane.plane import ToolPlaneAuthorizer  # noqa: PLC0415

        installed = self.registry.authorizer
        base = installed.base if isinstance(installed, _SessionToolAuthorizer) else installed
        selector = self.config.tool_selector
        plane_gate = ToolPlaneAuthorizer(selector) if isinstance(selector, ToolPlane) else None
        chain = chain_authorizers(base, plane_gate, SubagentToolAuthorizer())
        self.registry.set_authorizer(_SessionToolAuthorizer(base=base, chain=chain))

    async def initialize(self) -> None:
        if self._initialized:
            return
        # Hydrate session from store, or create a new one.
        store = self.config.stores.session
        if store is not None:
            existing = await store.get(self.id)
            if existing is None:
                await store.create(self.id, self.owner)
        await self.registry.initialize_mcp_servers()
        # Failed MCP servers are recorded on the registry; the session does not
        # raise on partial init. Consumers can inspect ``failed_mcp_servers``.
        if self.registry.failed_servers:
            import logging  # noqa: PLC0415

            log = logging.getLogger("agentkit.session")
            for name, msg in self.registry.failed_servers.items():
                log.warning("MCP server %r failed to initialize: %s", name, msg)
        self._initialized = True

    @property
    def failed_mcp_servers(self) -> dict[str, str]:
        """Map of MCP server name -> initialization error message.

        Empty before :meth:`initialize` runs and when all registered servers
        come up cleanly. Useful for surfacing MCP-server health to a UI.
        """
        return self.registry.failed_servers

    async def shutdown(self) -> None:
        await self.registry.shutdown()

    @asynccontextmanager
    async def run(self, user_text: str) -> AsyncGenerator[AsyncIterator[Event], None]:
        """Async-context that streams events for one turn.

        Usage:

            async with session.run("hello") as stream:
                async for ev in stream:
                    ...

        Leaving the ``async with`` block ends the turn: the Loop task is
        cancelled and whatever the turn produced is persisted, whether or not
        the stream was drained. A consumer that breaks out early therefore
        gets a recorded (and well-formed — see :meth:`_persist_turn`) partial
        turn rather than a detached background loop.
        """
        await self.initialize()
        queue: asyncio.Queue[Any] = asyncio.Queue(self.config.events.queue_size)

        # Build the turn context with the message history loaded — bounded,
        # tool-pair-safe, and repaired if an earlier turn left a dangling call.
        history: list[Message] = []
        load_notes: dict[str, Any] = {}
        store = self.config.stores.session
        if store is not None:
            history, load_notes = await self._load_history(store)

        user_msg = Message(
            id=new_id(MessageId),
            session_id=self.id,
            role=MessageRole.USER,
            content=[TextBlock(text=user_text)],
            created_at=datetime.now(UTC),
        )
        history.append(user_msg)
        if store is not None:
            await store.append_message(self.id, user_msg)

        ctx = TurnContext(
            session_id=self.id,
            turn_id=new_id(TurnId),
            call_id="",
            history=history,
            clock=SystemClock(),
            memory_store=self.config.stores.memory,
            memory_scope=self.config.stores.memory_scope,
            event_queue=queue,
        )
        ctx.metadata["owner"] = self.owner
        if load_notes:
            # Surfaced on the context so a host's on_iteration_start hook (or a
            # post-turn inspection) can see that this turn is running on a
            # shortened or repaired transcript.
            ctx.metadata["history_load"] = load_notes

        deps = self._build_deps()
        loop = Loop(
            ctx=ctx,
            handlers=self._handlers(),
            deps=deps,
            publish_phase_changed=self.config.events.publish_phase_changed,
        )
        # The user message is already in the store (appended above); the turn
        # owns everything after it.
        history_len_before_turn = len(history)

        runner = _TurnRunner(
            queue=queue,
            drain=lambda: self._drain_loop_into_queue(loop, queue),
            persist=lambda reason: self._persist_turn(
                ctx, history_len_before_turn, end_reason=reason
            ),
        )
        try:
            yield runner.stream()
        finally:
            await runner.finish()

    async def _load_history(self, store: SessionStore) -> tuple[list[Message], dict[str, Any]]:
        """Load the stored transcript for a new turn. Returns ``(history, notes)``.

        Three things the previous implicit ``list_messages()`` call did not do:

        1. The limit is explicit (:attr:`history_limit`), so the window is a
           decision rather than a store default.
        2. The cut is moved forward to a tool-pair-safe boundary, so a
           truncated load can never hand a provider a ``tool_result`` whose
           ``tool_use`` was dropped.
        3. Truncation is logged. Silently forgetting the first half of a long
           conversation is the failure mode this guards against.

        ``notes`` is empty when the load was clean, and otherwise carries what
        happened for :meth:`run` to stash on the TurnContext.
        """
        limit = max(1, self.history_limit)
        # Over-fetch by one: getting limit+1 back is the only way to know the
        # store had more than we are willing to load.
        fetched = await store.list_messages(self.id, limit=limit + 1)
        truncated = len(fetched) > limit
        start = _tool_safe_start(fetched, len(fetched) - limit if truncated else 0)
        history = fetched[start:]

        notes: dict[str, Any] = {}
        if truncated:
            notes = {"truncated": True, "limit": limit, "kept": len(history)}
            log.warning(
                "history_truncated",
                session_id=str(self.id),
                limit=limit,
                kept=len(history),
                dropped_from_window=start,
                hint="older messages were not loaded; compact the session or raise history_limit",
            )

        history, repaired = _repair_dangling_tool_uses(history, self.id)
        if repaired:
            notes["repaired_tool_uses"] = repaired
            log.warning(
                "dangling_tool_use_repaired",
                session_id=str(self.id),
                count=repaired,
                hint="stored transcript had tool_use blocks with no tool_result; "
                "synthesized cancelled results in memory for this turn",
            )
        return history, notes

    async def _persist_turn(
        self,
        ctx: TurnContext,
        start_index: int,
        *,
        end_reason: TurnEndReason | None,
    ) -> None:
        """Write everything the turn added to ``ctx.history`` to the store.

        Called exactly once per turn by :class:`_TurnRunner` — from ``run`` and
        from both resume paths, which is why a resumed turn's tool results and
        final assistant reply now reach the store at all.

        ``end_reason`` is the reason from the turn's ``TurnEnded`` event, or
        ``None`` when the consumer abandoned the stream before it arrived. Any
        ending other than a suspend gets a synthesized ``cancelled`` result for
        each unanswered ``ToolUseBlock``: persisting an assistant ``tool_use``
        with no matching ``tool_result`` corrupts the session permanently, as
        every later turn resends that malformed history. A suspend is the
        exception — its pending calls belong to the checkpoint and the resume
        will answer them for real.
        """
        store = self.config.stores.session
        if store is None:
            return
        new_messages = list(ctx.history[start_index:])

        # ``pending_user_approvals`` still populated means a checkpoint is out
        # there waiting to resolve these calls — true both for a clean
        # AWAITING_APPROVAL end and for a consumer that walked away before the
        # TurnEnded event reached it.
        suspended = end_reason is TurnEndReason.AWAITING_APPROVAL or bool(
            ctx.metadata.get("pending_user_approvals")
        )
        if not suspended:
            closers = [
                _cancelled_tool_result(self.id, tool_use_id, created_at=ctx.clock.now())
                for tool_use_id in _unresolved_tool_use_ids(ctx.history)
            ]
            if closers:
                log.warning(
                    "tool_calls_closed_as_cancelled",
                    session_id=str(self.id),
                    turn_id=str(ctx.turn_id),
                    count=len(closers),
                    end_reason=end_reason.value if end_reason is not None else None,
                )
                ctx.add_messages(closers)
                new_messages.extend(closers)

        for msg in new_messages:
            await store.append_message(self.id, msg)

    async def _load_resume_context(
        self, turn_id: TurnId, *, require_call_ids: Sequence[str] = ()
    ) -> tuple[TurnContext, asyncio.Queue[Any]]:
        """Load the checkpoint and rebuild a TurnContext for resumption.

        ``require_call_ids`` are validated against the checkpoint's pending
        approvals *before* the checkpoint is deleted. Order matters: a typo'd
        or already-resolved call_id used to destroy the checkpoint on its way
        to raising, leaving the suspended turn unresumable forever. Now the bad
        call raises :class:`CheckpointMissing` and the checkpoint survives for
        a corrected retry.

        Raises :class:`ApprovalTimeout` when the persisted ``approval_timeout_at``
        is in the past — preventing stale approvals from being honored after
        the configured window. For that case the checkpoint IS deleted, so a
        timed-out resume cannot be retried.
        """
        if self.config.stores.checkpoint is None:
            raise RuntimeError("CheckpointStore not configured; cannot resume")
        ckpt_id = CheckpointId(f"approval:{turn_id}")
        payload = await self.config.stores.checkpoint.load(ckpt_id)
        if payload is None:
            raise CheckpointMissing(ckpt_id)

        data = from_checkpoint_payload(payload)
        history = [Message.model_validate(m) for m in data["history"]]
        queue: asyncio.Queue[Any] = asyncio.Queue(self.config.events.queue_size)
        ctx = TurnContext(
            session_id=self.id,
            turn_id=turn_id,
            call_id="",
            history=history,
            clock=SystemClock(),
            memory_store=self.config.stores.memory,
            memory_scope=self.config.stores.memory_scope,
            event_queue=queue,
        )
        ctx.event_sequence = int(data.get("event_sequence", 0))
        # Taint is written into the checkpoint payload precisely so it survives
        # the suspend; a resumed turn that came back clean would re-enable every
        # write the guard denied before the user was even asked.
        ctx.tainted = bool(data.get("tainted", False))
        ctx.taint_sources = [TaintSource.model_validate(s) for s in data.get("taint_sources", [])]
        ctx.metadata.update(data.get("metadata", {}))
        ctx.metadata["owner"] = self.owner
        # Clear suspend marker so the orchestrator doesn't override end_reason.
        ctx.metadata.pop("suspend_reason", None)

        # Validate BEFORE the delete below — a rejected verdict must leave the
        # checkpoint intact.
        self._validate_call_ids(ctx, require_call_ids)

        # F24: enforce server-side approval timeout. Always delete the
        # checkpoint (success or timeout) so it can't be resumed later.
        timeout_iso = ctx.metadata.get("approval_timeout_at")
        await self.config.stores.checkpoint.delete(ckpt_id)
        if self._is_expired(timeout_iso):
            # The turn is now dead — checkpoint deleted, pending calls will
            # never run. Close them off in the store, or the stored
            # transcript keeps a ``tool_use`` that nothing can ever answer
            # and every later turn ships malformed history.
            raise await self._close_expired_checkpoint(ctx, turn_id, timeout_iso)
        return ctx, queue

    @staticmethod
    def _is_expired(timeout_iso: str | None) -> bool:
        """True when ``timeout_iso`` (an ``approval_timeout_at`` ISO string, or
        ``None`` for "no deadline recorded") is in the past."""
        if not timeout_iso:
            return False
        try:
            deadline = datetime.fromisoformat(timeout_iso)
        except ValueError:
            return False
        return datetime.now(UTC) > deadline

    async def _close_expired_checkpoint(
        self, ctx: TurnContext, turn_id: TurnId, timeout_iso: str | None
    ) -> ApprovalTimeout:
        """Strand ``ctx``'s pending calls, persist the turn as ERROR, and build
        the :class:`ApprovalTimeout` describing what expired.

        Shared by the resume path (which reaches here after a decision was
        already given, either way) and :meth:`check_approval_expiry` (which
        reaches here only after confirming the checkpoint IS expired — K11b:
        expiry used to be observable only when a client called resume, so an
        approval nobody ever answered emitted no event and wrote no audit
        row). Does not touch the checkpoint store itself; callers delete
        (or, for the resume path, already deleted) the checkpoint around this
        call.
        """
        stranded = [c["id"] for c in ctx.metadata.get("pending_user_approvals", [])]
        ctx.metadata["pending_user_approvals"] = []
        await self._persist_turn(ctx, len(ctx.history), end_reason=TurnEndReason.ERROR)
        return ApprovalTimeout(
            f"approval window for turn {turn_id} expired at {timeout_iso}",
            call_ids=stranded,
            # ctx.event_sequence is the checkpoint's next unused sequence
            # number — restarting the timeout stream at 0 would collide with
            # events the suspended turn already emitted under the same
            # turn_id (K11a).
            sequence_base=ctx.event_sequence,
        )

    async def check_approval_expiry(self, turn_id: TurnId) -> AsyncIterator[Event] | None:
        """Detect and close out an expired approval without a resume call.

        K11b: expiry was previously observable only through
        :meth:`resume_with_approval` — an approval nobody ever answers emits
        no event and writes no audit row, so "silence is not consent" held
        only if somebody eventually came back to check. This gives a host
        application a lazy-check primitive it can call from any access path
        that already knows about a pending turn — a periodic sweep over
        turn_ids the host tracked from its own persisted ``ApprovalNeeded``
        events, a status check when a session is reopened, and so on —
        without agentkit itself owning a scheduler or background-task
        lifecycle it does not otherwise have.

        Returns ``None`` and leaves the checkpoint untouched when there is
        nothing pending for ``turn_id``, or when it is pending but has not
        yet expired — this is a read *unless* it finds an expiry, never a
        destructive peek at a still-answerable approval. When the deadline
        has passed, deletes the checkpoint, persists the turn as
        ``TurnEndReason.ERROR`` (closing any dangling ``tool_use`` so the
        stored transcript stays well-formed), audits the resolution, and
        returns the same ``ApprovalResolved(expired=True)`` /
        ``Errored`` / ``TurnEnded`` stream :meth:`resume_with_approval` would
        have yielded, with sequence numbers continuing from the checkpoint's
        ``event_sequence`` (K11a) so a client dedupes correctly however the
        expiry was discovered.
        """
        if self.config.stores.checkpoint is None:
            return None
        ckpt_id = CheckpointId(f"approval:{turn_id}")
        payload = await self.config.stores.checkpoint.load(ckpt_id)
        if payload is None:
            return None

        data = from_checkpoint_payload(payload)
        metadata: dict[str, Any] = dict(data.get("metadata", {}))
        timeout_iso = metadata.get("approval_timeout_at")
        if not self._is_expired(timeout_iso):
            return None  # still pending, or no deadline recorded — untouched

        history = [Message.model_validate(m) for m in data["history"]]
        ctx = TurnContext(
            session_id=self.id,
            turn_id=turn_id,
            call_id="",
            history=history,
            clock=SystemClock(),
            memory_store=self.config.stores.memory,
            memory_scope=self.config.stores.memory_scope,
        )
        ctx.event_sequence = int(data.get("event_sequence", 0))
        ctx.metadata.update(metadata)
        ctx.metadata["owner"] = self.owner
        ctx.metadata.pop("suspend_reason", None)

        await self.config.stores.checkpoint.delete(ckpt_id)
        exc = await self._close_expired_checkpoint(ctx, turn_id, timeout_iso)
        return await self._approval_timeout_stream(turn_id, exc)

    @staticmethod
    def _validate_call_ids(ctx: TurnContext, call_ids: Sequence[str]) -> None:
        """Check every call_id is pending on ``ctx`` before anything is mutated.

        Raises :class:`CheckpointMissing` — the same exception
        :meth:`_apply_approval_decision` has always raised for an unknown
        call_id, kept so callers' existing error handling still applies.
        """
        pending_ids = {c["id"] for c in ctx.metadata.get("pending_user_approvals", [])}
        seen: set[str] = set()
        for call_id in call_ids:
            if call_id in seen:
                raise CheckpointMissing(f"call_id {call_id} given twice in one resume")
            seen.add(call_id)
            if call_id not in pending_ids:
                raise CheckpointMissing(f"call_id {call_id} not in pending approvals")

    @staticmethod
    def _deny_unresolved(ctx: TurnContext) -> list[dict[str, Any]]:
        """Auto-deny every approval the caller's verdicts did not cover.

        TOOL_EXECUTING runs only the approved/denied/unknown buckets, so a call
        left in ``pending_user_approvals`` would be dropped without ever
        producing a result — an invisible non-execution, and a dangling
        ``tool_use`` in the transcript. Denying it with an explicit reason
        makes the outcome visible to both the model and the consumer.

        Returns the synthetic verdicts, in ``resume_with_approval_batch``
        shape, so the caller can emit their ApprovalDenied events.
        """
        leftover = [c["id"] for c in ctx.metadata.get("pending_user_approvals", [])]
        verdicts: list[dict[str, Any]] = []
        for call_id in leftover:
            AgentSession._apply_approval_decision(
                ctx, call_id, "deny", None, _UNRESOLVED_DENY_REASON
            )
            verdicts.append(
                {
                    "call_id": call_id,
                    "decision": "deny",
                    "reason": _UNRESOLVED_DENY_REASON,
                    # Nobody ruled on this one; agentkit did, and the resolution
                    # event must not attribute it to the human who resumed.
                    "resolved_by": SYSTEM_RESOLVER,
                }
            )
        return verdicts

    @staticmethod
    def _apply_approval_decision(
        ctx: TurnContext,
        call_id: str,
        decision: str,
        edited_args: dict[str, Any] | None,
        reason: str | None,
    ) -> None:
        """Move the call from pending to approved/denied based on the verdict."""
        pending = list(ctx.metadata.get("pending_user_approvals", []))
        approved = list(ctx.metadata.get("approved_tool_calls", []))
        denied = list(ctx.metadata.get("denied_tool_calls", []))

        match = next((c for c in pending if c["id"] == call_id), None)
        if match is None:
            raise CheckpointMissing(f"call_id {call_id} not in pending approvals")
        pending.remove(match)
        if decision == "approve":
            if edited_args is not None:
                match = {**match, "arguments": edited_args}
            approved.append(match)
        else:
            denied.append({**match, "deny_reason": reason})

        ctx.metadata["pending_user_approvals"] = pending
        ctx.metadata["approved_tool_calls"] = approved
        ctx.metadata["denied_tool_calls"] = denied

    async def _approval_timeout_stream(
        self, turn_id: TurnId, exc: ApprovalTimeout
    ) -> AsyncIterator[Event]:
        """Surface a stale approval as one ApprovalResolved per stranded call,
        then Errored + TurnEnded(error), so consumers don't have to add a second
        exception path.

        A coroutine returning an iterator, rather than an async generator or a
        plain function: the audit rows must be written before the caller gets
        the iterator back, and only an awaited call can guarantee that. See the
        comment on the audit loop below.

        The per-call events matter more than the error: an expiring approval
        used to emit nothing addressed to the card the user is looking at, which
        is indistinguishable from a hang. ``expired=True`` with
        ``decision="deny"`` says the window closed and the call did not run.

        Sequence numbers continue from ``exc.sequence_base`` — the suspended
        turn's next unused ``event_sequence``, captured while its ``ctx`` was
        still in scope at the raise site — rather than restarting at 0. The
        turn_id is reused from the suspended turn, so restarting at 0 would
        collide with that turn's own events (``TurnStarted`` is always
        sequence 0), and a client that dedupes on ``(turn_id, sequence)``
        would silently drop the ApprovalResolved meant to close the card.
        """
        now = datetime.now(UTC)
        base = exc.sequence_base
        events: list[Event] = [
            ApprovalResolved(
                event_id=new_id(EventId),
                session_id=self.id,
                turn_id=turn_id,
                ts=now,
                sequence=base + i,
                call_id=call_id,
                decision="deny",
                resolved_by=SYSTEM_RESOLVER,
                reason=str(exc),
                expired=True,
            )
            for i, call_id in enumerate(exc.call_ids)
        ]
        events.append(
            Errored(
                event_id=new_id(EventId),
                session_id=self.id,
                turn_id=turn_id,
                ts=now,
                sequence=base + len(events),
                code=ErrorCode.APPROVAL_TIMEOUT,
                message=str(exc),
                recoverable=False,
            )
        )
        events.append(
            TurnEnded(
                event_id=new_id(EventId),
                session_id=self.id,
                turn_id=turn_id,
                ts=now,
                sequence=base + len(events),
                reason=TurnEndReason.ERROR,
                metrics=TurnMetrics(),
            )
        )

        # Audited here, in the coroutine, rather than inside ``_timeout_iter``:
        # an async generator body does not run until the first ``__anext__``,
        # so an audit loop placed at the top of the generator writes nothing
        # at all for a consumer that takes the iterator and never drains it.
        # ``resume_with_approval`` hands this stream out of an
        # ``@asynccontextmanager`` and a client that simply closes its approval
        # card — the single most likely thing to do on being told the window
        # shut — exits the ``async with`` without iterating. That path used to
        # produce zero audit rows while the comment above it asserted the
        # opposite. An expiry is a verdict the runtime reached on a human's
        # behalf; it goes on the record when the verdict is reached, not when
        # somebody gets around to looking at it.
        for ev in events:
            if isinstance(ev, ApprovalResolved):
                await record_audit(
                    self.audit_sink,
                    self._approval_audit_record(
                        turn_id,
                        ev.call_id,
                        # The context is gone with the checkpoint, so the
                        # tool name is genuinely unknown here.
                        tool_name="",
                        decision="deny",
                        edited_args=None,
                        reason=str(exc),
                        resolved_by=SYSTEM_RESOLVER,
                        expired=True,
                    ),
                )

        async def _timeout_iter() -> AsyncIterator[Event]:
            for ev in events:
                yield ev

        return _timeout_iter()

    def _approval_audit_record(
        self,
        turn_id: TurnId,
        call_id: str,
        *,
        tool_name: str,
        decision: str,
        edited_args: dict[str, Any] | None,
        reason: str | None,
        resolved_by: str,
        expired: bool,
    ) -> AuditRecord:
        """Build the audit record for one resolved approval.

        Consent is the half of a destructive action that a tool-call record
        cannot show: the dispatcher's record says the vault was deleted, this
        one says who agreed to it and whether they changed the arguments first.
        A verdict the runtime reached on its own is filed as agent activity —
        only a human's actual decision may be filed as :data:`SOURCE_HUMAN`.
        """
        approved = decision == "approve"
        detail: dict[str, Any] = {
            # Normalised the same way the session normalises it: anything but
            # exactly "approve" denied the call.
            "decision": "approve" if approved else "deny",
            "expired": expired,
        }
        if approved and edited_args is not None:
            preview, truncated = argument_preview(edited_args)
            detail["edited_args"] = preview
            detail["edited_args_truncated"] = truncated
        if not approved and reason:
            detail["reason"] = clean_text(reason, limit=_MAX_AUDIT_REASON_CHARS)
        return AuditRecord(
            ts=datetime.now(UTC),
            actor=resolved_by,
            action=ACTION_APPROVAL_RESOLVED,
            target=tool_name,
            detail=detail,
            source=SOURCE_AGENT if resolved_by == SYSTEM_RESOLVER else SOURCE_HUMAN,
            session_id=str(self.id),
            turn_id=str(turn_id),
            call_id=call_id,
        )

    @staticmethod
    def _tool_name_for_call(ctx: TurnContext, call_id: str) -> str:
        """The tool name behind ``call_id``, wherever the verdict left it.

        Searched across all three buckets because the caller runs after
        :meth:`_apply_approval_decision` has already moved the entry out of
        ``pending_user_approvals``.
        """
        for bucket in ("approved_tool_calls", "denied_tool_calls", "pending_user_approvals"):
            for entry in ctx.metadata.get(bucket, []):
                if entry.get("id") == call_id:
                    return str(entry.get("name", ""))
        return ""

    def _build_verdict_event(
        self,
        ctx: TurnContext,
        turn_id: TurnId,
        call_id: str,
        decision: str,
        edited_args: dict[str, Any] | None,
        reason: str | None,
    ) -> ApprovalGranted | ApprovalDenied:
        """Build the ApprovalGranted/ApprovalDenied event for a single verdict."""
        if decision == "approve":
            return ApprovalGranted(
                event_id=new_id(EventId),
                session_id=self.id,
                turn_id=turn_id,
                ts=datetime.now(UTC),
                sequence=ctx.next_sequence(),
                call_id=call_id,
                edited_args=edited_args,
            )
        return ApprovalDenied(
            event_id=new_id(EventId),
            session_id=self.id,
            turn_id=turn_id,
            ts=datetime.now(UTC),
            sequence=ctx.next_sequence(),
            call_id=call_id,
            reason=reason,
        )

    def _build_resolution_event(
        self,
        ctx: TurnContext,
        turn_id: TurnId,
        call_id: str,
        decision: str,
        edited_args: dict[str, Any] | None,
        reason: str | None,
        resolved_by: str,
    ) -> ApprovalResolved:
        """Build the ApprovalResolved event that pairs with a verdict.

        Emitted in addition to ApprovalGranted/ApprovalDenied rather than
        instead of them: those two are a published surface, and this one exists
        so a consumer has a single event type to close a card on — including on
        the expiry path, where no verdict event is produced at all.
        """
        approved = decision == "approve"
        return ApprovalResolved(
            event_id=new_id(EventId),
            session_id=self.id,
            turn_id=turn_id,
            ts=datetime.now(UTC),
            sequence=ctx.next_sequence(),
            call_id=call_id,
            # Mirror _apply_approval_decision: anything but exactly "approve"
            # denied the call, so report the denial rather than the input.
            decision="approve" if approved else "deny",
            resolved_by=resolved_by,
            edited_args=edited_args if approved else None,
            reason=None if approved else reason,
            expired=False,
        )

    async def _emit_verdict(
        self,
        queue: asyncio.Queue[Any],
        ctx: TurnContext,
        turn_id: TurnId,
        call_id: str,
        decision: str,
        edited_args: dict[str, Any] | None,
        reason: str | None,
        resolved_by: str,
    ) -> None:
        """Put the verdict pair (granted/denied, then resolved) on the queue,
        and write the decision to the audit sink."""
        await queue.put(
            self._build_verdict_event(ctx, turn_id, call_id, decision, edited_args, reason)
        )
        await queue.put(
            self._build_resolution_event(
                ctx, turn_id, call_id, decision, edited_args, reason, resolved_by
            )
        )
        await record_audit(
            self.audit_sink,
            self._approval_audit_record(
                turn_id,
                call_id,
                tool_name=self._tool_name_for_call(ctx, call_id),
                decision=decision,
                edited_args=edited_args,
                reason=reason,
                resolved_by=resolved_by,
                expired=False,
            ),
        )

    def _resumed_turn_runner(self, ctx: TurnContext, queue: asyncio.Queue[Any]) -> _TurnRunner:
        """Restart the Loop at TOOL_EXECUTING, draining into the verdict queue.

        Any further pending approvals will round-trip through approval_wait
        again as TOOL_EXECUTING -> TOOL_RESULTS.

        ``persist_from`` is the history length as rebuilt from the checkpoint:
        everything the resumed turn adds beyond that — the tool results and the
        final assistant reply — is new, and is what the resume path previously
        threw away. The checkpoint's own history was already persisted by the
        suspending turn, so nothing is written twice.
        """
        loop = Loop(
            ctx=ctx,
            handlers=self._handlers(),
            deps=self._build_deps(),
            starting_phase=Phase.TOOL_EXECUTING,
            publish_phase_changed=self.config.events.publish_phase_changed,
        )
        persist_from = len(ctx.history)
        return _TurnRunner(
            queue=queue,
            drain=lambda: self._drain_loop_into_queue(loop, queue),
            persist=lambda reason: self._persist_turn(ctx, persist_from, end_reason=reason),
        )

    @asynccontextmanager
    async def resume_with_approval(
        self,
        turn_id: TurnId,
        call_id: str,
        *,
        decision: str,  # "approve" | "deny"
        edited_args: dict[str, Any] | None = None,
        reason: str | None = None,
        resolved_by: str | None = None,
    ) -> AsyncGenerator[AsyncIterator[Event], None]:
        """Resume a suspended turn with the user's approval verdict.

        Looks up the checkpoint persisted by approval_wait, enforces the
        ``approval_timeout_seconds`` deadline (raising :class:`ApprovalTimeout`
        when stale), applies the decision, and restarts the Loop at
        TOOL_EXECUTING. Emits :class:`ApprovalGranted` or :class:`ApprovalDenied`
        as the first event in the resumed stream so consumers can render
        the verdict (including any ``edited_args``) in their UI.

        Each verdict is also reported as an :class:`ApprovalResolved`, which is
        the event a consumer should close its approval card on — it is the only
        one that also fires when the approval expires.

        ``resolved_by`` names who ruled; it defaults to the session owner, which
        is right for a single-user session and wrong for a shared one, so a
        transport that knows the actual clicker should pass it.

        An unknown ``call_id`` raises :class:`CheckpointMissing` and leaves the
        checkpoint intact, so the turn stays resumable.

        For a turn that suspended on multiple pending tool calls, prefer
        :meth:`resume_with_approval_batch`: this method rules on one call, and
        every other pending call is auto-denied with the reason
        ``"not resolved in this resume"`` (each surfacing its own
        :class:`ApprovalDenied` event). That is deliberate — leaving them
        pending would drop them silently in TOOL_EXECUTING — but it is a
        blunt outcome the batch API lets the user avoid.
        """
        await self.initialize()
        try:
            ctx, queue = await self._load_resume_context(turn_id, require_call_ids=[call_id])
        except ApprovalTimeout as exc:
            yield await self._approval_timeout_stream(turn_id, exc)
            return

        self._apply_approval_decision(ctx, call_id, decision, edited_args, reason)
        # Emit the verdict events before any further work so consumers see them
        # at the top of the resumed stream.
        who = resolved_by or str(self.owner)
        await self._emit_verdict(queue, ctx, turn_id, call_id, decision, edited_args, reason, who)
        for auto in self._deny_unresolved(ctx):
            await self._emit_verdict(
                queue,
                ctx,
                turn_id,
                auto["call_id"],
                "deny",
                None,
                auto["reason"],
                auto["resolved_by"],
            )
        runner = self._resumed_turn_runner(ctx, queue)
        try:
            yield runner.stream()
        finally:
            await runner.finish()

    @asynccontextmanager
    async def resume_with_approval_batch(
        self,
        turn_id: TurnId,
        decisions: list[dict[str, Any]],
        *,
        resolved_by: str | None = None,
    ) -> AsyncGenerator[AsyncIterator[Event], None]:
        """Resume a suspended turn after applying a batch of approval verdicts.

        Like :meth:`resume_with_approval`, but applies every verdict in
        ``decisions`` to the checkpoint before restarting the Loop once. A UI
        can present a single approval card covering N pending tool calls and
        resume them all on one verdict, instead of forcing a
        resume -> suspend -> resume round-trip per call.

        Each entry is ``{"call_id": str, "decision": "approve" | "deny",
        "edited_args"?: dict | None, "reason"?: str | None,
        "resolved_by"?: str}``. Decisions are applied — and their
        ApprovalGranted/ApprovalDenied verdict plus :class:`ApprovalResolved`
        emitted — in list order, at the head of the resumed stream. An entry's
        ``resolved_by`` overrides the call-level one, which defaults to the
        session owner.

        Batching is required for correctness, not just ergonomics: TOOL_EXECUTING
        runs only the approved/denied/unknown buckets, so any call left in
        ``pending_user_approvals`` after a resume would be dropped. Every
        pending call is therefore resolved before the loop advances — those the
        batch does not mention are auto-denied with the reason
        ``"not resolved in this resume"``.

        An unknown or duplicated ``call_id`` raises :class:`CheckpointMissing`
        before anything is applied, and leaves the checkpoint intact so the
        caller can retry with a corrected batch.
        """
        await self.initialize()
        try:
            ctx, queue = await self._load_resume_context(
                turn_id, require_call_ids=[d["call_id"] for d in decisions]
            )
        except ApprovalTimeout as exc:
            yield await self._approval_timeout_stream(turn_id, exc)
            return

        # Apply every verdict first so a bad call_id fails fast before any tool
        # runs, then emit the verdict events at the head of the resumed stream.
        for d in decisions:
            self._apply_approval_decision(
                ctx,
                d["call_id"],
                d["decision"],
                d.get("edited_args"),
                d.get("reason"),
            )
        who = resolved_by or str(self.owner)
        emitted = [*decisions, *self._deny_unresolved(ctx)]
        for d in emitted:
            await self._emit_verdict(
                queue,
                ctx,
                turn_id,
                d["call_id"],
                d["decision"],
                d.get("edited_args"),
                d.get("reason"),
                d.get("resolved_by") or who,
            )
        runner = self._resumed_turn_runner(ctx, queue)
        try:
            yield runner.stream()
        finally:
            await runner.finish()

    async def _drain_loop_into_queue(self, loop: Loop, queue: asyncio.Queue[Any]) -> None:
        async for ev in loop.run():
            await queue.put(ev)

    def _handlers(self) -> dict[Phase, Any]:
        return {
            Phase.INTENT_GATE: handle_intent_gate,
            Phase.CONTEXT_BUILD: handle_context_build,
            Phase.STREAMING: handle_streaming,
            Phase.TOOL_PHASE: handle_tool_phase,
            Phase.APPROVAL_WAIT: handle_approval_wait,
            Phase.TOOL_EXECUTING: handle_tool_executing,
            Phase.TOOL_RESULTS: handle_tool_results,
            Phase.FINALIZE_CHECK: handle_finalize_check,
            Phase.MEMORY_EXTRACT: handle_memory_extract,
        }

    def _build_deps(self) -> dict[str, Any]:
        gc = self.config.guards
        lc = self.config.loop
        deps: dict[str, Any] = {
            "provider": self.provider,
            "provider_selector": self.config.provider_selector,
            "model_selector": self.config.model_selector,
            "tool_selector": self.config.tool_selector,
            "on_iteration_start": self.config.on_iteration_start,
            "message_builder": MessageBuilder(
                model=self.model,
                max_tokens=4096,
                # Stamp session_id so providers can route it to per-session
                # tracing (see agentkit._stream_trace, opt-in via env). No
                # effect on the request wire — adapters strip ``metadata``
                # before sending.
                metadata={"session_id": str(self.id)},
            ),
            "registry": self.registry,
            "system_blocks": self.system_blocks,
            # effective_intent_gate(), not gc.intent: it composes the built-in
            # per-principal rate limiter ahead of any consumer gate, and caches
            # the composed gate so the sliding window survives the per-turn
            # rebuild of these deps.
            "intent_gate": gc.effective_intent_gate(),
            "approval_gate": gc.approval or RiskBasedApprovalGate(),
            "dispatcher": ToolDispatcher(
                registry=self.registry,
                policy=DispatchPolicy(max_parallel=self.config.tool_dispatch.max_parallel),
                audit=self.audit_sink,
            ),
            # Also in deps so a subagent's dispatcher — and any handler that
            # needs to record something — resolves the same sink as the parent
            # instead of quietly defaulting to the no-op one.
            "audit_sink": self.audit_sink,
            "finalize_validator": gc.finalize or StructuralFinalizeValidator(),
            "success_claim": gc.success_claim if gc.success_claim_enabled else None,
            "approval_timeout_seconds": gc.approval_timeout_seconds,
            "max_finalize_retries": lc.max_finalize_retries,
            "max_missing_finalize_reprompts": lc.max_missing_finalize_reprompts,
            "force_finalize_on_missing_reprompt": lc.force_finalize_on_missing_reprompt,
            "max_iterations": lc.max_iterations,
            "max_consecutive_tool_errors": lc.max_consecutive_tool_errors,
            "max_stream_retries": lc.max_stream_retries,
            "stream_retry_base_delay_seconds": lc.stream_retry_base_delay_seconds,
            "max_claim_corrections": lc.max_claim_corrections,
            "streaming_chunk_timeout_seconds": lc.streaming_chunk_timeout_seconds,
            "checkpoint_store": self.config.stores.checkpoint,
        }
        # SubagentDispatcher needs deps to construct its child loops; building
        # it here closes the loop (the child Loop reuses the parent's deps,
        # so a nested kit.subagent.spawn works to ``max_depth``).
        deps["subagent_dispatcher"] = SubagentDispatcher(
            deps=deps,
            max_depth=self.config.loop.max_subagent_depth,
        )
        return deps
