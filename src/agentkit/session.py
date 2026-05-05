"""AgentSession — top-level entry point for consumers.

Wraps a single conversation: persistent SessionStore-backed history, plus a
``run(user_message)`` async-iterator API that yields events. Internally
constructs a Loop per turn.
"""

from __future__ import annotations

import asyncio
import contextlib
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from agentkit._content import TextBlock
from agentkit._ids import CheckpointId, EventId, MessageId, OwnerId, SessionId, TurnId, new_id
from agentkit._messages import Message, MessageRole
from agentkit.errors import ApprovalTimeout, CheckpointMissing
from agentkit.events import (
    ApprovalDenied,
    ApprovalGranted,
    ErrorCode,
    Errored,
    Event,
    TurnEnded,
    TurnEndReason,
    TurnMetrics,
)
from agentkit.guards.approval import RiskBasedApprovalGate
from agentkit.guards.finalize import RuleBasedFinalizeValidator
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
    from collections.abc import AsyncGenerator, AsyncIterator

    from agentkit.config import AgentConfig
    from agentkit.providers.base import Provider, SystemBlock
    from agentkit.tools.registry import ToolRegistry


class AgentSession:
    """One conversation. Run turns via ``async with session.run(text) as stream:``."""

    def __init__(
        self,
        *,
        owner: OwnerId,
        config: AgentConfig,
        provider: Provider,
        registry: ToolRegistry,
        model: str,
        system_blocks: list[SystemBlock] | None = None,
        session_id: SessionId | None = None,
    ) -> None:
        self.owner = owner
        self.id: SessionId = session_id or new_id(SessionId)
        self.config = config
        self.provider = provider
        self.registry = registry
        self.model = model
        self.system_blocks = system_blocks or []
        self._initialized = False

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
        """
        await self.initialize()
        queue: asyncio.Queue[Any] = asyncio.Queue(self.config.events.queue_size)

        # Build the turn context with full message history loaded.
        history: list[Message] = []
        store = self.config.stores.session
        if store is not None:
            history = await store.list_messages(self.id)

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
            event_queue=queue,
        )
        ctx.metadata["owner"] = self.owner

        deps = self._build_deps()
        loop = Loop(ctx=ctx, handlers=self._handlers(), deps=deps)
        history_len_before_turn = len(history)

        async def _iter() -> AsyncIterator[Event]:
            run_task = asyncio.create_task(self._drain_loop_into_queue(loop, queue))
            try:
                while True:
                    item = await queue.get()
                    yield item
                    if isinstance(item, TurnEnded):
                        break
            finally:
                if not run_task.done():
                    run_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await run_task
                # Persist any new assistant messages produced during the turn.
                if store is not None:
                    for msg in ctx.history[history_len_before_turn:]:
                        await store.append_message(self.id, msg)

        try:
            yield _iter()
        finally:
            pass

    async def _load_resume_context(self, turn_id: TurnId) -> tuple[TurnContext, asyncio.Queue[Any]]:
        """Load the checkpoint and rebuild a TurnContext for resumption.

        Raises :class:`ApprovalTimeout` when the persisted ``approval_timeout_at``
        is in the past — preventing stale approvals from being honored after
        the configured window. The checkpoint is deleted regardless so a
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
            event_queue=queue,
        )
        ctx.metadata.update(data.get("metadata", {}))
        ctx.metadata["owner"] = self.owner
        # Clear suspend marker so the orchestrator doesn't override end_reason.
        ctx.metadata.pop("suspend_reason", None)

        # F24: enforce server-side approval timeout. Always delete the
        # checkpoint (success or timeout) so it can't be resumed later.
        timeout_iso = ctx.metadata.get("approval_timeout_at")
        await self.config.stores.checkpoint.delete(ckpt_id)
        if timeout_iso:
            try:
                deadline = datetime.fromisoformat(timeout_iso)
            except ValueError:
                deadline = None
            if deadline is not None and datetime.now(UTC) > deadline:
                raise ApprovalTimeout(
                    f"approval window for turn {turn_id} expired at {timeout_iso}"
                )
        return ctx, queue

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

    @asynccontextmanager
    async def resume_with_approval(
        self,
        turn_id: TurnId,
        call_id: str,
        *,
        decision: str,  # "approve" | "deny"
        edited_args: dict[str, Any] | None = None,
        reason: str | None = None,
    ) -> AsyncGenerator[AsyncIterator[Event], None]:
        """Resume a suspended turn with the user's approval verdict.

        Looks up the checkpoint persisted by approval_wait, enforces the
        ``approval_timeout_seconds`` deadline (raising :class:`ApprovalTimeout`
        when stale), applies the decision, and restarts the Loop at
        TOOL_EXECUTING. Emits :class:`ApprovalGranted` or :class:`ApprovalDenied`
        as the first event in the resumed stream so consumers can render
        the verdict (including any ``edited_args``) in their UI.
        """
        await self.initialize()
        try:
            ctx, queue = await self._load_resume_context(turn_id)
        except ApprovalTimeout as exc:
            # Surface the timeout as a single Errored + TurnEnded(error) stream
            # so consumers don't have to add a second exception path.
            now = datetime.now(UTC)
            errored = Errored(
                event_id=new_id(EventId),
                session_id=self.id,
                turn_id=turn_id,
                ts=now,
                sequence=0,
                code=ErrorCode.APPROVAL_TIMEOUT,
                message=str(exc),
                recoverable=False,
            )
            ended = TurnEnded(
                event_id=new_id(EventId),
                session_id=self.id,
                turn_id=turn_id,
                ts=now,
                sequence=1,
                reason=TurnEndReason.ERROR,
                metrics=TurnMetrics(),
            )

            async def _timeout_iter() -> AsyncIterator[Event]:
                yield errored
                yield ended

            try:
                yield _timeout_iter()
            finally:
                pass
            return

        self._apply_approval_decision(ctx, call_id, decision, edited_args, reason)

        # Emit the verdict event before any further work so consumers see it
        # at the top of the resumed stream.
        sequence = ctx.metadata.get("event_sequence", 200)
        verdict_event: ApprovalGranted | ApprovalDenied
        if decision == "approve":
            verdict_event = ApprovalGranted(
                event_id=new_id(EventId),
                session_id=self.id,
                turn_id=turn_id,
                ts=datetime.now(UTC),
                sequence=sequence,
                call_id=call_id,
                edited_args=edited_args,
            )
        else:
            verdict_event = ApprovalDenied(
                event_id=new_id(EventId),
                session_id=self.id,
                turn_id=turn_id,
                ts=datetime.now(UTC),
                sequence=sequence,
                call_id=call_id,
                reason=reason,
            )
        ctx.metadata["event_sequence"] = sequence + 1
        await queue.put(verdict_event)

        # Restart the Loop at TOOL_EXECUTING. Any further pending approvals
        # will round-trip through approval_wait again as TOOL_EXECUTING -> TOOL_RESULTS.
        loop = Loop(
            ctx=ctx,
            handlers=self._handlers(),
            deps=self._build_deps(),
            starting_phase=Phase.TOOL_EXECUTING,
        )

        async def _iter() -> AsyncIterator[Event]:
            run_task = asyncio.create_task(self._drain_loop_into_queue(loop, queue))
            try:
                while True:
                    item = await queue.get()
                    yield item
                    if isinstance(item, TurnEnded):
                        break
            finally:
                if not run_task.done():
                    run_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await run_task

        try:
            yield _iter()
        finally:
            pass

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
        deps: dict[str, Any] = {
            "provider": self.provider,
            "message_builder": MessageBuilder(
                model=self.model,
                max_tokens=4096,
            ),
            "registry": self.registry,
            "system_blocks": self.system_blocks,
            "intent_gate": gc.intent,
            "approval_gate": gc.approval or RiskBasedApprovalGate(),
            "dispatcher": ToolDispatcher(
                registry=self.registry,
                policy=DispatchPolicy(max_parallel=self.config.tool_dispatch.max_parallel),
            ),
            "finalize_validator": gc.finalize or RuleBasedFinalizeValidator(),
            "success_claim": gc.success_claim if gc.success_claim_enabled else None,
            "approval_timeout_seconds": gc.approval_timeout_seconds,
            "max_finalize_retries": self.config.loop.max_finalize_retries,
            "max_iterations": self.config.loop.max_iterations,
            "max_consecutive_tool_errors": self.config.loop.max_consecutive_tool_errors,
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
