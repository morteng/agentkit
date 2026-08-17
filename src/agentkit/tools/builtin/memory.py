"""The memory tools: save, recall, search, list and forget.

Operate on the MemoryStore + MemoryScope attached to the TurnContext by the Loop.
If the store/scope are not configured, returns an error result (no silent no-op).

The ``key`` argument is model-authored, so it is validated here before it
reaches any store — see :func:`agentkit.store.redis.keys.validate_memory_key`.
The Redis backend escapes keys, so a hostile key is already *harmless*; this
check is about giving the model a readable, retryable error instead of letting
an empty or absurdly long key become an unreachable row. The store validates
again on its own write path (defence in depth, and other tools may write too).

WHY SEARCH, LIST AND FORGET EXIST (added 2026-08-17)
----------------------------------------------------
``MemoryStore`` is a five-method protocol and only ``save`` and ``recall`` were
ever wrapped as tools. ``search``, ``list_keys`` and ``delete`` were implemented
by **every** backend — Redis, the fake, and at least one downstream SQLite store
— and reachable by nothing. That is not a missing feature so much as a feature
delivered with three of its five doors bricked up, and the consequence was
specific rather than theoretical:

* ``recall`` takes an **exact key**. A model in a later session has no way to
  guess the key string a model in an earlier session invented, so cross-session
  recall failed even for facts that had been saved correctly. Memory that cannot
  be found again is storage, not memory.
* With no ``list``, "what do you remember about me?" had no answer available to
  the model except "nothing" — indistinguishable from an empty store, which is
  exactly how the one real deployment of this presented for weeks.
* With no ``delete``, a person could ask an agent to remember something and had
  no way to ask it to stop. Forgetting should never be harder than remembering.

``delete`` is deliberately ``LOW_WRITE`` and not ``DESTRUCTIVE``: hosts commonly
gate DESTRUCTIVE behind an administrator, and "only an admin may delete your
memory of me" inverts who the control is for. It is also why forgetting is
exposed as ``kit.memory.forget`` rather than ``…delete`` — the word a person
would use, in a tool name that reaches user-facing surfaces.
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


#: Ceiling on how many hits one search may return. The model chooses `limit`,
#: and an unbounded one turns a search into "load the entire store into the
#: context window" — cheap to ask for, expensive to answer, and it degrades the
#: result it was meant to improve.
_MAX_SEARCH_LIMIT = 25

MEMORY_SEARCH_SPEC = ToolSpec(
    name="kit.memory.search",
    description=(
        "Find previously saved facts by searching their text. Use this rather "
        "than guessing a key: keys are chosen when a fact is saved and are not "
        "predictable later. Returns the matching facts themselves."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Words to look for."},
            "limit": {
                "type": "integer",
                "description": f"How many to return (1-{_MAX_SEARCH_LIMIT}).",
                "default": 10,
            },
        },
        "required": ["query"],
    },
    returns=None,
    risk=RiskLevel.READ,
    idempotent=True,
    side_effects=SideEffects.NONE,
    requires_approval=ApprovalPolicy.NEVER,
    cache_ttl_seconds=None,
    timeout_seconds=5.0,
)


MEMORY_LIST_SPEC = ToolSpec(
    name="kit.memory.list",
    description=(
        "List the keys of everything saved for the current person. Use this to "
        "answer questions about what is remembered, and to find the exact key "
        "to recall or forget."
    ),
    parameters={"type": "object", "properties": {}},
    returns=None,
    risk=RiskLevel.READ,
    idempotent=True,
    side_effects=SideEffects.NONE,
    requires_approval=ApprovalPolicy.NEVER,
    cache_ttl_seconds=None,
    timeout_seconds=5.0,
)


MEMORY_FORGET_SPEC = ToolSpec(
    name="kit.memory.forget",
    description=(
        "Delete one saved fact by key. Use when asked to forget something. "
        "Find the key with search or list first — this does nothing useful "
        "against a guessed key."
    ),
    parameters={
        "type": "object",
        "properties": {"key": {"type": "string"}},
        "required": ["key"],
    },
    returns=None,
    risk=RiskLevel.LOW_WRITE,
    # Deleting the same key twice leaves the same state, and the handler reports
    # honestly which of the two happened rather than claiming a deletion it did
    # not perform.
    idempotent=True,
    side_effects=SideEffects.LOCAL,
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


async def memory_search_handler(args: dict[str, Any], ctx: TurnContext) -> ToolResult:
    if (err := _need_store(ctx)) is not None:
        return err
    assert ctx.memory_store is not None and ctx.memory_scope is not None
    # No _check_key here: a query is prose, not a key, and validate_memory_key
    # would reject perfectly good searches (spaces, punctuation, length).
    query = str(args.get("query", ""))
    limit = max(1, min(int(args.get("limit", 10)), _MAX_SEARCH_LIMIT))
    hits = await ctx.memory_store.search(ctx.memory_scope, query, limit=limit)
    if not hits:
        return ToolResult(
            call_id=ctx.call_id,
            status="ok",
            content=[ContentBlockOut(type="text", text="no matches")],
            error=None,
            duration_ms=0,
            cached=False,
        )
    # The key travels with the text because it is the handle for `forget` and
    # `recall`. A search that returned only prose would leave the model able to
    # find a fact and unable to act on it.
    lines = [f"{hit.key}: {hit.value.text}" for hit in hits]
    return ToolResult(
        call_id=ctx.call_id,
        status="ok",
        content=[ContentBlockOut(type="text", text="\n".join(lines))],
        error=None,
        duration_ms=0,
        cached=False,
    )


async def memory_list_handler(args: dict[str, Any], ctx: TurnContext) -> ToolResult:
    if (err := _need_store(ctx)) is not None:
        return err
    assert ctx.memory_store is not None and ctx.memory_scope is not None
    keys = await ctx.memory_store.list_keys(ctx.memory_scope)
    # "nothing saved yet" rather than an empty string: an empty result and a
    # broken tool must not look the same to the model, because the difference
    # is what it tells the person.
    text = "\n".join(sorted(keys)) if keys else "nothing saved yet"
    return ToolResult(
        call_id=ctx.call_id,
        status="ok",
        content=[ContentBlockOut(type="text", text=text)],
        error=None,
        duration_ms=0,
        cached=False,
    )


async def memory_forget_handler(args: dict[str, Any], ctx: TurnContext) -> ToolResult:
    if (err := _need_store(ctx)) is not None:
        return err
    key = str(args["key"])
    if (err := _check_key(key, ctx)) is not None:
        return err
    assert ctx.memory_store is not None and ctx.memory_scope is not None
    # `delete` is silent about whether anything was there, so the existence
    # check happens here. Saying "forgotten" about a key that never existed
    # would have the agent confirm a change it did not make — the receipt has
    # to describe what happened, not what was asked for.
    existed = await ctx.memory_store.recall(ctx.memory_scope, key) is not None
    await ctx.memory_store.delete(ctx.memory_scope, key)
    text = f"forgot {key}" if existed else f"nothing saved under {key}"
    return ToolResult(
        call_id=ctx.call_id,
        status="ok",
        content=[ContentBlockOut(type="text", text=text)],
        error=None,
        duration_ms=0,
        cached=False,
    )
