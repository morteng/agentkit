import asyncio
from datetime import UTC, datetime, timedelta

import pytest
import structlog

from agentkit._content import TextBlock
from agentkit._ids import MessageId, SessionId, new_id
from agentkit._messages import Message, MessageRole
from agentkit.guards.intent import (
    ContentBlocklistCheck,
    DefaultIntentGate,
    InMemoryRateLimitCheck,
    MaxMessageLengthCheck,
    verified_principal,
)
from agentkit.loop.context import TurnContext


def _user(text: str) -> Message:
    from datetime import UTC, datetime

    return Message(
        id=new_id(MessageId),
        session_id=new_id(SessionId),
        role=MessageRole.USER,
        content=[TextBlock(text=text)],
        created_at=datetime.now(UTC),
    )


@pytest.mark.asyncio
async def test_no_checks_allows_everything():
    gate = DefaultIntentGate(checks=[])
    ctx = TurnContext.empty()
    ctx.add_message(_user("hi"))
    decision = await gate.evaluate(ctx)
    assert decision.allow is True


@pytest.mark.asyncio
async def test_max_length_rejects_oversize_message():
    gate = DefaultIntentGate(checks=[MaxMessageLengthCheck(max_chars=10)])
    ctx = TurnContext.empty()
    ctx.add_message(_user("x" * 100))
    decision = await gate.evaluate(ctx)
    assert decision.allow is False
    assert "max" in (decision.reason or "").lower()


@pytest.mark.asyncio
async def test_blocklist_rejects_match():
    gate = DefaultIntentGate(checks=[ContentBlocklistCheck(patterns=[r"forbidden"])])
    ctx = TurnContext.empty()
    ctx.add_message(_user("this contains a forbidden phrase"))
    decision = await gate.evaluate(ctx)
    assert decision.allow is False


@pytest.mark.asyncio
async def test_rate_limit_after_threshold():
    check = InMemoryRateLimitCheck(turns_per_minute=2)
    gate = DefaultIntentGate(checks=[check])
    ctx = TurnContext.empty()
    ctx.add_message(_user("hi"))
    ctx.metadata["owner"] = "u1"
    assert (await gate.evaluate(ctx)).allow
    assert (await gate.evaluate(ctx)).allow
    assert not (await gate.evaluate(ctx)).allow


@pytest.mark.asyncio
async def test_rate_limit_buckets_by_verified_owner_not_by_session():
    """A burst funnelled through fresh sessions (or spawned subagents, which get
    a fresh session_id but inherit ``owner``) counts against one bucket."""
    check = InMemoryRateLimitCheck(turns_per_minute=2)
    first = TurnContext.empty()
    first.metadata["owner"] = "u1"
    second = TurnContext.empty()  # different session_id
    second.metadata["owner"] = "u1"

    assert (await check.evaluate(first)).allow
    assert (await check.evaluate(second)).allow
    assert not (await check.evaluate(second)).allow


@pytest.mark.asyncio
async def test_rate_limit_isolates_distinct_owners():
    check = InMemoryRateLimitCheck(turns_per_minute=1)
    a = TurnContext.empty()
    a.metadata["owner"] = "u1"
    b = TurnContext.empty()
    b.metadata["owner"] = "u2"

    assert (await check.evaluate(a)).allow
    assert not (await check.evaluate(a)).allow
    assert (await check.evaluate(b)).allow, "one principal must not exhaust another's budget"


@pytest.mark.asyncio
async def test_rate_limit_falls_back_to_session_when_unowned():
    """No owner: per-session buckets, not one shared 'anon' bucket that would
    let any session lock out every other one."""
    check = InMemoryRateLimitCheck(turns_per_minute=1)
    a = TurnContext.empty()
    b = TurnContext.empty()

    assert (await check.evaluate(a)).allow
    assert not (await check.evaluate(a)).allow
    assert (await check.evaluate(b)).allow


def test_verified_principal_prefers_owner():
    ctx = TurnContext.empty()
    assert verified_principal(ctx) == f"session:{ctx.session_id}"
    ctx.metadata["owner"] = "u1"
    assert verified_principal(ctx) == "owner:u1"


@pytest.mark.asyncio
async def test_rate_limit_accepts_a_custom_principal_key():
    check = InMemoryRateLimitCheck(
        turns_per_minute=1,
        principal_key=lambda ctx: str(ctx.metadata.get("tenant", "?")),
    )
    a = TurnContext.empty()
    a.metadata["tenant"] = "t1"
    b = TurnContext.empty()
    b.metadata["tenant"] = "t1"

    assert (await check.evaluate(a)).allow
    assert not (await check.evaluate(b)).allow


@pytest.mark.asyncio
async def test_rate_limit_rejection_is_logged_and_carries_retry_after():
    check = InMemoryRateLimitCheck(turns_per_minute=1)
    ctx = TurnContext.empty()
    ctx.metadata["owner"] = "u1"
    await check.evaluate(ctx)

    with structlog.testing.capture_logs() as logs:
        decision = await check.evaluate(ctx)

    assert decision.allow is False
    assert decision.metadata["limit"] == 1
    # metadata is str -> object, so narrow before comparing. The isinstance is
    # part of the assertion: a retry hint the caller cannot compare against a
    # number is not a usable retry hint.
    retry_after = decision.metadata["retry_after_seconds"]
    assert isinstance(retry_after, int | float)
    assert retry_after >= 1
    assert [e for e in logs if e["event"] == "rate_limit_exceeded"]


@pytest.mark.asyncio
async def test_window_expiry_lets_the_next_turn_through():
    check = InMemoryRateLimitCheck(turns_per_minute=1, window_seconds=0.05)
    ctx = TurnContext.empty()
    ctx.metadata["owner"] = "u1"

    assert (await check.evaluate(ctx)).allow
    assert not (await check.evaluate(ctx)).allow
    await asyncio.sleep(0.06)
    assert (await check.evaluate(ctx)).allow


def test_stale_buckets_are_pruned():
    """Unbounded growth would be a slow leak in a process that runs for months:
    the ``session:`` fallback key alone adds one entry per session."""
    check = InMemoryRateLimitCheck(turns_per_minute=10)
    now = datetime.now(UTC)
    for i in range(10):
        check._window[f"owner:stale{i}"].append(now - timedelta(minutes=5))
    check._window["owner:live"].append(now)

    check._prune(now - timedelta(seconds=60))

    assert list(check._window) == ["owner:live"]
