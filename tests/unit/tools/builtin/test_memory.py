import pytest

from agentkit.loop.context import TurnContext
from agentkit.store.fakes import FakeMemoryStore
from agentkit.store.memory import MemoryScope
from agentkit.tools.builtin.memory import memory_recall_handler, memory_save_handler


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
