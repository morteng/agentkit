import pytest

from agentkit.store.fakes import FakeCheckpointStore  # used as KV; bytes payloads
from agentkit.tools.cache import ToolResultCache, cache_key
from agentkit.tools.spec import ContentBlockOut, ToolResult


def test_cache_key_stable_across_arg_orderings():
    a = cache_key("kit.x", {"a": 1, "b": 2})
    b = cache_key("kit.x", {"b": 2, "a": 1})
    assert a == b


def test_cache_key_changes_with_args():
    assert cache_key("kit.x", {"a": 1}) != cache_key("kit.x", {"a": 2})


@pytest.mark.asyncio
async def test_cache_store_and_load_round_trip():
    backing = FakeCheckpointStore()  # any bytes KV works
    cache = ToolResultCache(backing)
    res = ToolResult(
        call_id="c1",
        status="ok",
        content=[ContentBlockOut(type="text", text="hi")],
        error=None,
        duration_ms=5,
        cached=False,
    )
    key = cache_key("kit.x", {"a": 1})
    await cache.store(key, res, ttl_seconds=300)
    loaded = await cache.load(key)
    assert loaded is not None
    assert loaded.content[0].text == "hi"
    assert loaded.cached is True  # marker flipped on load
