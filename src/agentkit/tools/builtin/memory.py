"""kit.memory.save and kit.memory.recall.

Operate on the MemoryStore + MemoryScope attached to the TurnContext by the Loop.
If the store/scope are not configured, returns an error result (no silent no-op).

The ``key`` argument is model-authored, so it is validated here before it
reaches any store — see :func:`agentkit.store.redis.keys.validate_memory_key`.
The Redis backend escapes keys, so a hostile key is already *harmless*; this
check is about giving the model a readable, retryable error instead of letting
an empty or absurdly long key become an unreachable row. The store validates
again on its own write path (defence in depth, and other tools may write too).
"""

from datetime import UTC, datetime
from typing import Any

from agentkit.loop.context import TurnContext
from agentkit.store.memory import MemoryValue
from agentkit.store.redis.keys import validate_memory_key
from agentkit.tools.spec import (
    ApprovalPolicy,
    ContentBlockOut,
    RiskLevel,
    SideEffects,
    ToolError,
    ToolResult,
    ToolSpec,
)

MEMORY_SAVE_SPEC = ToolSpec(
    name="kit.memory.save",
    description="Save a fact for later recall in this or future sessions.",
    parameters={
        "type": "object",
        "properties": {
            "key": {"type": "string"},
            "text": {"type": "string"},
            "tags": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["key", "text"],
    },
    returns=None,
    risk=RiskLevel.LOW_WRITE,
    idempotent=True,
    side_effects=SideEffects.LOCAL,
    requires_approval=ApprovalPolicy.NEVER,
    cache_ttl_seconds=None,
    timeout_seconds=5.0,
)


MEMORY_RECALL_SPEC = ToolSpec(
    name="kit.memory.recall",
    description="Recall a previously saved fact by key.",
    parameters={
        "type": "object",
        "properties": {"key": {"type": "string"}},
        "required": ["key"],
    },
    returns=None,
    risk=RiskLevel.READ,
    idempotent=True,
    side_effects=SideEffects.NONE,
    requires_approval=ApprovalPolicy.NEVER,
    cache_ttl_seconds=None,
    timeout_seconds=5.0,
)


def _need_store(ctx: TurnContext) -> ToolResult | None:
    if ctx.memory_store is None or ctx.memory_scope is None:
        return ToolResult(
            call_id=ctx.call_id,
            status="error",
            content=[],
            error=ToolError(
                code="memory_not_configured",
                message="No MemoryStore on this session.",
            ),
            duration_ms=0,
            cached=False,
        )
    return None


def _check_key(key: str, ctx: TurnContext) -> ToolResult | None:
    """Reject an unusable memory key with a model-readable error."""
    try:
        validate_memory_key(key)
    except ValueError as exc:
        return ToolResult(
            call_id=ctx.call_id,
            status="error",
            content=[],
            error=ToolError(
                code="invalid_memory_key",
                message=str(exc),
                retryable=True,
            ),
            duration_ms=0,
            cached=False,
        )
    return None


async def memory_save_handler(args: dict[str, Any], ctx: TurnContext) -> ToolResult:
    if (err := _need_store(ctx)) is not None:
        return err
    if (err := _check_key(str(args["key"]), ctx)) is not None:
        return err
    now = datetime.now(UTC)
    value = MemoryValue(
        text=str(args["text"]),
        tags=list(args.get("tags", [])),
        created_at=now,
        updated_at=now,
    )
    assert ctx.memory_store is not None and ctx.memory_scope is not None
    await ctx.memory_store.save(ctx.memory_scope, str(args["key"]), value)
    return ToolResult(
        call_id=ctx.call_id,
        status="ok",
        content=[ContentBlockOut(type="text", text=f"saved {args['key']}")],
        error=None,
        duration_ms=0,
        cached=False,
    )


async def memory_recall_handler(args: dict[str, Any], ctx: TurnContext) -> ToolResult:
    if (err := _need_store(ctx)) is not None:
        return err
    if (err := _check_key(str(args["key"]), ctx)) is not None:
        return err
    assert ctx.memory_store is not None and ctx.memory_scope is not None
    value = await ctx.memory_store.recall(ctx.memory_scope, str(args["key"]))
    if value is None:
        return ToolResult(
            call_id=ctx.call_id,
            status="ok",
            content=[ContentBlockOut(type="text", text="not found")],
            error=None,
            duration_ms=0,
            cached=False,
        )
    return ToolResult(
        call_id=ctx.call_id,
        status="ok",
        content=[ContentBlockOut(type="text", text=value.text)],
        error=None,
        duration_ms=0,
        cached=False,
    )
