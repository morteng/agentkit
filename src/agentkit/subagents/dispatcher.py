"""SubagentDispatcher — spawn a nested Loop and return the assistant's final text.

The spawn request is **model-authored**: ``kit.subagent.spawn`` takes a
``prompt``, a ``tools`` allowlist and a free-form ``context`` dict straight
from the provider's tool call. Everything in this module is written on that
assumption:

* ``context`` may not carry runtime security metadata at all — a spawn naming
  any of :data:`~agentkit.subagents.isolation.RESERVED_CONTEXT_KEYS` is
  rejected outright, and the trusted values are stamped after the merge so
  that ordering is belt to the rejection's braces.
* ``tools`` is enforced, not recorded: the child loop runs against a
  :class:`~agentkit.subagents.isolation.RestrictedToolRegistry` view that
  default-denies anything unlisted, and can only ever narrow what the parent
  already had.
* The child cannot suspend for approval. It has no user and no resume path,
  so :class:`~agentkit.subagents.isolation.NestedApprovalGate` turns
  ``NEEDS_USER`` into an auto-deny, the child's checkpoint store is removed so
  no orphan checkpoint can be written, and a child that somehow still ends
  suspended raises :class:`SubagentApprovalRequired` instead of silently
  returning a truncated answer.
* Taint is anti-laundering in **both** directions. ``fresh_child_context``
  already stops a tainted parent from spawning a falsely-clean child (see
  ``subagents/isolation.py``); :meth:`SubagentDispatcher.spawn` closes the
  other direction so a clean parent cannot come back falsely-clean after a
  child it spawned reads untrusted content. Propagation runs from a
  ``finally``, so it fires whether the child finished, raised, or ended
  suspended on an approval it could never resolve — a subagent cannot buy the
  parent a false all-clear by failing instead of returning.
"""

import asyncio
import contextlib
from collections.abc import AsyncIterator, Sequence
from typing import TYPE_CHECKING, Any

from agentkit._content import Provenance, TextBlock
from agentkit._logging import get_logger
from agentkit._messages import MessageRole
from agentkit.errors import AgentkitError
from agentkit.events import TextDelta, ToolCallStarted
from agentkit.events.base import BaseEvent
from agentkit.guards.taint import TaintSource, is_tainted, mark_taint, taint_sources
from agentkit.loop.context import TurnContext
from agentkit.loop.handlers.approval_wait import handle_approval_wait
from agentkit.loop.handlers.context_build import handle_context_build
from agentkit.loop.handlers.finalize_check import handle_finalize_check
from agentkit.loop.handlers.intent_gate import handle_intent_gate
from agentkit.loop.handlers.memory_extract import handle_memory_extract
from agentkit.loop.handlers.streaming import handle_streaming
from agentkit.loop.handlers.tool_executing import handle_tool_executing
from agentkit.loop.handlers.tool_phase import handle_tool_phase
from agentkit.loop.handlers.tool_results import handle_tool_results
from agentkit.loop.orchestrator import Loop
from agentkit.loop.phase import Phase
from agentkit.loop.tool_dispatcher import DispatchPolicy, ToolDispatcher
from agentkit.subagents.isolation import (
    NestedApprovalGate,
    RestrictedToolRegistry,
    effective_allowlist,
    fresh_child_context,
    reserved_keys_in,
)
from agentkit.tools.spec import ToolResult

if TYPE_CHECKING:
    from agentkit.tools.registry import ToolRegistry

log = get_logger(__name__)

# Buffer subagent text deltas this many characters before flushing as a
# parent-stream ToolCallProgress. Newlines also flush. Tuned so the parent
# UI sees a sentence-or-two cadence rather than a flood of single-char events.
_SUBAGENT_TEXT_FLUSH_AT = 80

# Put on the child's event queue by the pump task once ``loop.run()`` is
# exhausted (or has failed). Consumers stop there. A sentinel rather than
# "stop at TurnEnded" so a loop that dies before emitting TurnEnded cannot
# leave the consumer waiting on a queue nothing will ever fill again.
_END_OF_CHILD_STREAM = object()


class SubagentDepthExceeded(AgentkitError):
    pass


class SubagentContextRejected(AgentkitError):
    """A spawn's ``context`` tried to set runtime-owned security metadata."""


class SubagentApprovalRequired(AgentkitError):
    """A subagent turn ended suspended on human approval, which cannot resolve."""


async def _surface_subagent_events(parent: TurnContext, events: AsyncIterator[BaseEvent]) -> None:
    """Drain a child loop's events and surface key milestones as
    :class:`ToolCallProgress` on the parent's event queue.

    Two milestones reach the parent:
      * Child :class:`TextDelta` is debounced into chunks (flushed on newline
        or after ``_SUBAGENT_TEXT_FLUSH_AT`` characters) and emitted as
        ``"subagent: <text>"``.
      * Child :class:`ToolCallStarted` is emitted as
        ``"subagent calling <tool_name>"``.

    All other child events (PhaseChanged, ApprovalNeeded, MessageStarted,
    etc.) are consumed but not surfaced; they are internal to the child loop.

    ``events`` must be the **merged** child stream — see
    :func:`_child_event_stream`. ``Loop.run()`` on its own yields only
    lifecycle events (TurnStarted / PhaseChanged / TurnEnded); TextDelta and
    ToolCallStarted are put on ``ctx.event_queue`` by the streaming handler,
    so filtering ``loop.run()`` alone can never match anything.
    """
    buffer: list[str] = []

    async def _flush() -> None:
        if not buffer:
            return
        text = "".join(buffer).strip()
        buffer.clear()
        if text:
            await parent.report_tool_progress(f"subagent: {text}")

    async for ev in events:
        if isinstance(ev, TextDelta):
            buffer.append(ev.delta)
            joined = "".join(buffer)
            if "\n" in joined or len(joined) >= _SUBAGENT_TEXT_FLUSH_AT:
                await _flush()
        elif isinstance(ev, ToolCallStarted):
            await _flush()
            await parent.report_tool_progress(f"subagent calling {ev.tool_name}")
    await _flush()


async def _pump_loop_into_queue(loop: Loop, queue: "asyncio.Queue[Any]") -> None:
    """Forward ``loop.run()``'s lifecycle events onto the child's event queue.

    Same shape as :meth:`AgentSession._drain_loop_into_queue`: the handlers
    already push their own events onto ``ctx.event_queue``, so pushing the
    orchestrator's events onto the *same* queue is what merges the two streams
    into one totally ordered sequence.

    ``queue`` must be unbounded: the sentinel is enqueued from ``finally``,
    including on cancellation, where awaiting a bounded put could not complete.
    :meth:`SubagentDispatcher.spawn` gives every child an unbounded queue.
    """
    try:
        async for ev in loop.run():
            await queue.put(ev)
    finally:
        queue.put_nowait(_END_OF_CHILD_STREAM)


async def _child_event_stream(queue: "asyncio.Queue[Any]") -> AsyncIterator[BaseEvent]:
    """Yield every child event — handler-emitted and orchestrator-emitted alike."""
    while True:
        item = await queue.get()
        if item is _END_OF_CHILD_STREAM:
            return
        if isinstance(item, BaseEvent):
            yield item


class SubagentDispatcher:
    """Runs one nested Loop per :meth:`spawn`, isolated from the parent turn.

    ``deps`` is the parent's dependency bag. It is never handed to the child
    unchanged: :meth:`_child_deps` swaps in the restricted registry, a
    dispatcher bound to it, the nested approval gate, and a nested
    SubagentDispatcher so a grandchild inherits the narrowed set rather than
    reaching back through to the parent's full one.
    """

    def __init__(self, *, deps: dict[str, Any], max_depth: int = 3) -> None:
        self._deps = deps
        self._max_depth = max_depth

    async def spawn(
        self,
        parent: TurnContext,
        *,
        prompt: str,
        tools: Sequence[str],
        extra_context: dict[str, Any],
    ) -> str:
        depth = int(parent.metadata.get("subagent_depth", 0)) + 1
        if depth > self._max_depth:
            raise SubagentDepthExceeded(f"max subagent depth {self._max_depth} exceeded")

        # Reject before doing any work. The merge order below would already
        # defeat an overwrite attempt, but a rejection is auditable and cannot
        # be undone by a later refactor that reorders two statements.
        offending = reserved_keys_in(extra_context)
        if offending:
            log.warning(
                "subagent_context_rejected",
                keys=offending,
                depth=depth,
                session_id=str(parent.session_id),
            )
            raise SubagentContextRejected(
                f"subagent context may not set runtime-owned keys: {', '.join(offending)}. "
                "These are stamped by the runtime (recursion depth, owner, tool allowlist, "
                "capability tiers) and are not addressable from a spawn request."
            )

        parent_registry: ToolRegistry = self._deps["registry"]
        allowed = effective_allowlist(parent_registry, tools)

        child = fresh_child_context(parent, prompt=prompt)
        # Untrusted first...
        child.metadata.update(extra_context)
        # ...trusted last, so nothing in ``extra_context`` can shadow it.
        self._stamp_trusted_metadata(child, parent, depth=depth, allowed=allowed)
        child.event_queue = asyncio.Queue()

        child_deps = self._child_deps(allowed)
        loop = Loop(ctx=child, handlers=_CHILD_HANDLERS, deps=child_deps)

        try:
            await self._run_child(parent, loop, child.event_queue)
            self._check_no_pending_approvals(child)
            await self._surface_denied_approvals(parent, child)

            # Extract the final assistant message text.
            for msg in reversed(child.history):
                if msg.role is MessageRole.ASSISTANT:
                    texts = [b.text for b in msg.content if isinstance(b, TextBlock)]
                    if texts:
                        return "\n".join(texts)
            return ""
        finally:
            # Runs whether the block above returned, raised (including
            # SubagentApprovalRequired below), or the child loop itself
            # blew up — taint must reach the parent regardless of how this
            # call ends, or a subagent that errors out after reading a
            # malicious page buys the parent a false all-clear.
            self._propagate_taint_from_child(parent, child)

    @staticmethod
    def _propagate_taint_from_child(parent: TurnContext, child: TurnContext) -> None:
        """Anti-laundering, the other direction: mark ``parent`` for whatever
        untrusted content ``child`` ingested during its own run.

        ``fresh_child_context`` already copies the parent's *pre-existing*
        taint down onto a fresh child (see ``subagents/isolation.py``) — that
        is the half that was implemented. This is the other half: a child
        started clean (or already tainted by inheritance) can *itself* ingest
        untrusted content via its own tool calls, and that must latch back
        onto ``parent`` before control returns to it, or the parent's turn
        continues believing it never read anything untrusted.

        Goes through :func:`~agentkit.guards.taint.mark_taint` — the same
        function every ordinary tool call is marked through — rather than
        setting ``parent.tainted`` directly, so this stays the one place the
        latch-and-record logic lives. One synthetic, zero-content
        :class:`~agentkit.tools.spec.ToolResult` is built per child
        :class:`~agentkit.guards.taint.TaintSource` and fed through it,
        carrying the source's own ``call_id`` and ``tool_name`` forward so an
        ``ApprovalNeeded`` card on the parent's turn can still name the real
        tool (``web.fetch``, three levels down) rather than a bare
        ``kit.subagent.spawn``. ``mark_taint`` dedupes by value, so a source
        the parent already recorded (inherited taint, echoed back by the
        child) is not written twice.

        Fails closed: if the child's taint state cannot even be read, or it
        is tainted but recorded no source at all (e.g. restored from a
        checkpoint written before sources existed), the parent is still
        latched, with a fallback source naming ``kit.subagent.spawn`` itself
        — a denial must always have a cause to show, never a bare "tainted".
        """
        try:
            child_tainted = is_tainted(child)
            sources = taint_sources(child) if child_tainted else []
        except Exception:
            log.warning(
                "subagent_taint_state_unreadable",
                session_id=str(parent.session_id),
            )
            child_tainted, sources = True, []

        if not child_tainted:
            return

        if not sources:
            sources = [
                TaintSource(
                    call_id=f"subagent:{parent.call_id or parent.turn_id}",
                    tool_name="kit.subagent.spawn",
                    kind=Provenance.UNTRUSTED.value,
                )
            ]

        for source in sources:
            synthetic = ToolResult(
                call_id=source.call_id,
                status="ok",
                content=[],
                error=None,
                duration_ms=0,
                cached=False,
                provenance=Provenance.UNTRUSTED,
            )
            mark_taint(parent, synthetic, tool_name=source.tool_name)

    # ---- Wiring ------------------------------------------------------------

    @staticmethod
    def _stamp_trusted_metadata(
        child: TurnContext,
        parent: TurnContext,
        *,
        depth: int,
        allowed: frozenset[str],
    ) -> None:
        """Write the runtime-owned metadata onto the child, after the merge.

        ``owner`` and the capability keys are inherited from the parent rather
        than defaulted: a child acts as the same principal with, at most, the
        same capabilities. A parent that does not carry a key leaves the child
        without it too — inheritance never invents authority.
        """
        child.metadata["subagent_depth"] = depth
        child.metadata["allowed_tools"] = sorted(allowed)
        for key in ("owner", "tier_overrides", "capabilities", "discovered_tools"):
            if key in parent.metadata:
                child.metadata[key] = parent.metadata[key]
            else:
                child.metadata.pop(key, None)

    def _child_deps(self, allowed: frozenset[str]) -> dict[str, Any]:
        parent_registry: ToolRegistry = self._deps["registry"]
        view = (
            parent_registry.restrict(allowed)
            if isinstance(parent_registry, RestrictedToolRegistry)
            else RestrictedToolRegistry(inner=parent_registry, allowed=allowed)
        )

        child_deps = {
            **self._deps,
            "registry": view,
            # The parent's ToolDispatcher holds a reference to the *parent's*
            # registry; reusing it would route every child call around the
            # restriction. Rebuild it against the view, preserving the
            # session's configured parallelism — and its audit sink, or a
            # subagent would be the one place tool calls happen unrecorded.
            "dispatcher": ToolDispatcher(
                registry=view,
                policy=self._dispatch_policy(),
                audit=self._deps.get("audit_sink"),
            ),
            "approval_gate": NestedApprovalGate(self._deps.get("approval_gate")),
            # No user is attached to a subagent turn, so no checkpoint it could
            # write would ever be resumed. Removing the store means a suspend
            # cannot leave an orphan behind at all.
            "checkpoint_store": None,
        }
        # Self-reference, same shape as AgentSession._build_deps: a grandchild
        # must be dispatched from the *child's* deps so its registry view is
        # the already-restricted one.
        child_deps["subagent_dispatcher"] = SubagentDispatcher(
            deps=child_deps, max_depth=self._max_depth
        )
        return child_deps

    def _dispatch_policy(self) -> DispatchPolicy:
        """The parent dispatcher's policy, or the default if it has none.

        Package-private attribute access, deliberately: ToolDispatcher exposes
        no accessor and a child running at a different parallelism than the
        session was configured for would be a surprise.
        """
        policy = getattr(self._deps.get("dispatcher"), "_policy", None)
        return policy if isinstance(policy, DispatchPolicy) else DispatchPolicy()

    # ---- Execution ---------------------------------------------------------

    @staticmethod
    async def _run_child(parent: TurnContext, loop: Loop, queue: "asyncio.Queue[Any]") -> None:
        """Drive the child loop, surfacing its merged event stream to ``parent``."""
        pump = asyncio.create_task(_pump_loop_into_queue(loop, queue))
        try:
            await _surface_subagent_events(parent, _child_event_stream(queue))
            # The pump has already queued its sentinel by the time the stream
            # ends; awaiting it re-raises anything the loop itself raised.
            await pump
        finally:
            if not pump.done():
                pump.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await pump

    @staticmethod
    def _check_no_pending_approvals(child: TurnContext) -> None:
        """Fail loudly if the child ended suspended on approval.

        :class:`NestedApprovalGate` converts NEEDS_USER to an auto-deny, so
        reaching this normally means a consumer supplied an approval gate the
        dispatcher did not wrap, or a handler enqueued approvals directly.
        Either way the turn is unresumable — returning the child's partial
        text as if it were an answer is the failure the audit flagged.
        """
        pending: list[dict[str, Any]] = child.metadata.get("pending_user_approvals") or []
        if not pending:
            return
        names = ", ".join(str(call.get("name", "?")) for call in pending)
        log.error(
            "subagent_suspended_on_approval",
            session_id=str(child.session_id),
            turn_id=str(child.turn_id),
            tools=names,
        )
        raise SubagentApprovalRequired(
            f"subagent required approval; not permitted in nested context "
            f"(tools: {names}). Run these tools from the parent agent, where "
            f"the user can answer the approval prompt."
        )

    @staticmethod
    async def _surface_denied_approvals(parent: TurnContext, child: TurnContext) -> None:
        """Tell the parent stream about calls the nested gate auto-denied."""
        denied: list[dict[str, Any]] = child.metadata.get("subagent_denied_approvals") or []
        for call in denied:
            await parent.report_tool_progress(
                f"subagent denied {call.get('name', '?')}: approval is not available "
                f"inside a subagent"
            )


_CHILD_HANDLERS = {
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
