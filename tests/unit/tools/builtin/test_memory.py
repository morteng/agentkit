import pytest

from agentkit.loop.context import TurnContext
from agentkit.store.fakes import FakeMemoryStore
from agentkit.store.memory import MemoryScope
from agentkit.tools.builtin import DEFAULT_BUILTINS
from agentkit.tools.builtin.memory import (
    _MAX_SEARCH_LIMIT,
    memory_forget_handler,
    memory_list_handler,
    memory_recall_handler,
    memory_save_handler,
    memory_search_handler,
)
from agentkit.tools.spec import RiskLevel


@pytest.mark.asyncio
async def test_memory_save_and_recall_round_trip():
    store = FakeMemoryStore()
    scope = MemoryScope(namespace="t", user_id="u1")
    ctx = TurnContext.empty(call_id="c1", memory_store=store, memory_scope=scope)

    await memory_save_handler({"key": "k1", "text": "user lives in Oslo"}, ctx)
    res = await memory_recall_handler({"key": "k1"}, ctx)
    assert res.status == "ok"
    assert "Oslo" in (res.content[0].text or "")


@pytest.mark.asyncio
async def test_memory_recall_missing_returns_not_found():
    store = FakeMemoryStore()
    ctx = TurnContext.empty(
        call_id="c1",
        memory_store=store,
        memory_scope=MemoryScope(namespace="t"),
    )
    res = await memory_recall_handler({"key": "nope"}, ctx)
    assert res.status == "ok"
    assert "not found" in (res.content[0].text or "").lower()


@pytest.mark.asyncio
async def test_memory_handlers_error_without_scope():
    """A store but no scope is still not configured — explicit error, no silent no-op."""
    ctx = TurnContext.empty(call_id="c1", memory_store=FakeMemoryStore(), memory_scope=None)

    for res in (
        await memory_save_handler({"key": "k1", "text": "v"}, ctx),
        await memory_recall_handler({"key": "k1"}, ctx),
        await memory_search_handler({"query": "v"}, ctx),
        await memory_list_handler({}, ctx),
        await memory_forget_handler({"key": "k1"}, ctx),
    ):
        assert res.status == "error"
        assert res.error is not None
        assert res.error.code == "memory_not_configured"


@pytest.mark.asyncio
async def test_memory_handlers_error_without_store():
    """Symmetric case: a scope but no store."""
    ctx = TurnContext.empty(
        call_id="c1", memory_store=None, memory_scope=MemoryScope(namespace="t")
    )

    for res in (
        await memory_save_handler({"key": "k1", "text": "v"}, ctx),
        await memory_recall_handler({"key": "k1"}, ctx),
        await memory_search_handler({"query": "v"}, ctx),
        await memory_list_handler({}, ctx),
        await memory_forget_handler({"key": "k1"}, ctx),
    ):
        assert res.status == "error"
        assert res.error is not None
        assert res.error.code == "memory_not_configured"


# ---------- search / list / forget ----------
#
# These three wrap store methods that every backend already implemented and
# nothing could reach. The tests below are written against the *seam that was
# missing* — "can a later turn find a fact whose key it does not know", not
# "does the store round-trip", which was already covered and already passing
# while the capability was unusable.


def _ctx(store: FakeMemoryStore, scope: MemoryScope) -> TurnContext:
    return TurnContext.empty(call_id="c1", memory_store=store, memory_scope=scope)


@pytest.mark.asyncio
async def test_a_later_turn_finds_a_fact_without_knowing_its_key():
    """The whole point. `recall` needs the exact key; nothing invents that twice."""
    store, scope = FakeMemoryStore(), MemoryScope(namespace="t", user_id="u1")
    await memory_save_handler(
        {"key": "beverage-pref-2026", "text": "drinks oat milk, not dairy"},
        _ctx(store, scope),
    )

    # A fresh turn, and a key nobody would guess.
    res = await memory_search_handler({"query": "oat milk"}, _ctx(store, scope))
    assert res.status == "ok"
    text = res.content[0].text or ""
    assert "oat milk" in text
    # The key travels with the hit, because it is the handle for forget/recall.
    assert "beverage-pref-2026" in text


@pytest.mark.asyncio
async def test_search_with_no_matches_says_so():
    store, scope = FakeMemoryStore(), MemoryScope(namespace="t")
    await memory_save_handler({"key": "k1", "text": "lives in Oslo"}, _ctx(store, scope))
    res = await memory_search_handler({"query": "Bergen"}, _ctx(store, scope))
    assert res.status == "ok"
    assert "no matches" in (res.content[0].text or "")


@pytest.mark.asyncio
async def test_search_limit_is_clamped_not_honoured_blindly():
    """A model-chosen limit must not become 'put the whole store in the context'."""
    store, scope = FakeMemoryStore(), MemoryScope(namespace="t")
    for i in range(_MAX_SEARCH_LIMIT + 10):
        await memory_save_handler({"key": f"k{i}", "text": "coffee"}, _ctx(store, scope))

    res = await memory_search_handler({"query": "coffee", "limit": 9999}, _ctx(store, scope))
    lines = (res.content[0].text or "").splitlines()
    assert len(lines) == _MAX_SEARCH_LIMIT

    # And the other end: 0 or negative must not mean "none" or crash.
    res = await memory_search_handler({"query": "coffee", "limit": 0}, _ctx(store, scope))
    assert len((res.content[0].text or "").splitlines()) == 1


@pytest.mark.asyncio
async def test_list_distinguishes_empty_from_broken():
    """'nothing saved yet' and a failure must not look the same to the model."""
    store, scope = FakeMemoryStore(), MemoryScope(namespace="t")

    res = await memory_list_handler({}, _ctx(store, scope))
    assert res.status == "ok"
    assert "nothing saved yet" in (res.content[0].text or "")

    await memory_save_handler({"key": "b", "text": "second"}, _ctx(store, scope))
    await memory_save_handler({"key": "a", "text": "first"}, _ctx(store, scope))
    res = await memory_list_handler({}, _ctx(store, scope))
    assert (res.content[0].text or "").splitlines() == ["a", "b"]


@pytest.mark.asyncio
async def test_forget_removes_the_fact_and_search_stops_finding_it():
    store, scope = FakeMemoryStore(), MemoryScope(namespace="t")
    await memory_save_handler({"key": "k1", "text": "allergic to penicillin"}, _ctx(store, scope))

    res = await memory_forget_handler({"key": "k1"}, _ctx(store, scope))
    assert res.status == "ok"
    assert "forgot k1" in (res.content[0].text or "")

    # Gone from both retrieval paths, not just the one we deleted through.
    assert "not found" in (
        (await memory_recall_handler({"key": "k1"}, _ctx(store, scope))).content[0].text or ""
    )
    assert "no matches" in (
        (await memory_search_handler({"query": "penicillin"}, _ctx(store, scope))).content[0].text
        or ""
    )


@pytest.mark.asyncio
async def test_forget_does_not_claim_a_deletion_it_did_not_make():
    """The receipt describes what happened, not what was asked for."""
    store, scope = FakeMemoryStore(), MemoryScope(namespace="t")
    res = await memory_forget_handler({"key": "never-existed"}, _ctx(store, scope))
    assert res.status == "ok"
    text = (res.content[0].text or "").lower()
    assert "nothing saved under" in text
    assert "forgot" not in text


@pytest.mark.asyncio
async def test_search_query_is_not_validated_as_a_key():
    """Prose queries have spaces and punctuation; key validation would reject them."""
    store, scope = FakeMemoryStore(), MemoryScope(namespace="t")
    await memory_save_handler({"key": "k1", "text": "prefers window seats"}, _ctx(store, scope))
    res = await memory_search_handler({"query": "window seats, please!"}, _ctx(store, scope))
    assert res.status == "ok"
    assert res.error is None


@pytest.mark.asyncio
async def test_forget_still_validates_its_key():
    store, scope = FakeMemoryStore(), MemoryScope(namespace="t")
    res = await memory_forget_handler({"key": ""}, _ctx(store, scope))
    assert res.status == "error"
    assert res.error is not None
    assert res.error.code == "invalid_memory_key"


def test_the_three_new_tools_are_registered_by_default():
    """Registered, or none of the above reaches a model.

    The bug this file exists to close was never in the handlers — `search` was
    correct in every backend. It was that nothing wrapped them, and a handler
    test would pass either way.
    """
    by_name = {spec.name: spec for spec, _ in DEFAULT_BUILTINS}
    assert "kit.memory.search" in by_name
    assert "kit.memory.list" in by_name
    assert "kit.memory.forget" in by_name


def test_risk_levels_keep_reading_available_to_a_read_only_role():
    """Asking what is remembered must not require write permission.

    Hosts gate on risk (guest→READ). If listing were a write, the role least
    able to consent to being remembered would also be the one unable to check.
    """
    by_name = {spec.name: spec for spec, _ in DEFAULT_BUILTINS}
    assert by_name["kit.memory.search"].risk is RiskLevel.READ
    assert by_name["kit.memory.list"].risk is RiskLevel.READ
    # Forgetting is a write, but only LOW_WRITE: making it DESTRUCTIVE would
    # put "delete what you know about me" behind an administrator.
    assert by_name["kit.memory.forget"].risk is RiskLevel.LOW_WRITE
