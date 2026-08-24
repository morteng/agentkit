"""Provenance survives the memory write path — protocol, fake and Redis backends.

The defect these cover: ``save()`` dropped the trust classification, so
``recall()`` handed back untrusted third-party text with the same standing as a
fact the host wrote itself. Recalled memories are injected as context, so that
is a laundering path through the taint guard, not a cosmetic omission.

The tests are written against the *round trip* — what a later read can see —
rather than against the argument going in, because a store that accepted the
keyword and discarded it would pass the second and still be broken.
"""

import inspect
import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

import pytest
from redis.asyncio import Redis

from agentkit._content import Provenance
from agentkit.store.fakes import FakeMemoryStore
from agentkit.store.memory import (
    MemoryScope,
    MemoryStore,
    MemoryValue,
    stamp_provenance,
)
from agentkit.store.redis.keys import KeyBuilder
from agentkit.store.redis.memory import RedisMemoryStore

if TYPE_CHECKING:
    from agentkit.store.redis.client import RedisClient

SCOPE = MemoryScope(namespace="t", user_id="u1")


def _value(text: str = "a fact", **kw: Any) -> MemoryValue:
    now = datetime.now(UTC)
    return MemoryValue(text=text, created_at=now, updated_at=now, **kw)


# ---------- MemoryValue.provenance: the read-side surface ----------


def test_memory_value_defaults_to_system_so_existing_construction_is_unchanged():
    assert _value().provenance is Provenance.SYSTEM


def test_a_payload_written_before_the_field_existed_still_validates():
    """No migration: rows already in a store have no `provenance` key at all."""
    legacy = {
        "text": "written last year",
        "payload": {},
        "tags": [],
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
    }
    restored = MemoryValue.model_validate(legacy)
    assert restored.provenance is Provenance.SYSTEM


def test_provenance_is_serialised_so_a_json_backend_can_persist_it():
    dumped = _value(provenance=Provenance.UNTRUSTED).model_dump(mode="json")
    assert dumped["provenance"] == "untrusted"


# ---------- stamp_provenance: what the default actually means ----------


def test_the_default_keeps_the_label_the_value_already_carries():
    """Distinguishes "the default applied" from "the argument was passed".

    A literal ``Provenance.SYSTEM`` default would pass every test that only
    ever writes SYSTEM values — and would silently launder this one.
    """
    marked = _value(provenance=Provenance.UNTRUSTED)
    assert stamp_provenance(marked, None).provenance is Provenance.UNTRUSTED


def test_an_explicit_argument_overrides_the_value():
    assert stamp_provenance(_value(), Provenance.UNTRUSTED).provenance is Provenance.UNTRUSTED


def test_stamping_changes_nothing_else_about_the_value():
    original = _value("remember the milk")
    stamped = stamp_provenance(original, Provenance.PRINCIPAL)
    assert stamped.text == original.text
    assert stamped.created_at == original.created_at
    assert stamped.updated_at == original.updated_at


# ---------- the protocol ----------


def test_save_provenance_is_keyword_only_with_a_default():
    """Both properties are the backwards-compatibility contract.

    Positional would break every existing call site; no default would break
    every existing caller.
    """
    param = inspect.signature(MemoryStore.save).parameters["provenance"]
    assert param.kind is inspect.Parameter.KEYWORD_ONLY
    assert param.default is None


def test_an_implementation_written_before_provenance_still_satisfies_the_protocol():
    """`MemoryStore` is @runtime_checkable and consumers isinstance() against it.

    A required argument here would have made every third-party store fail a
    check it passes today — which is why the issue asks for a keyword default.
    """

    class LegacyStore:
        async def save(self, scope: MemoryScope, key: str, value: MemoryValue) -> None: ...

        async def recall(self, scope: MemoryScope, key: str) -> MemoryValue | None: ...

        async def search(self, scope: MemoryScope, query: str, *, limit: int = 10) -> list[Any]: ...

        async def list_keys(self, scope: MemoryScope) -> list[str]: ...

        async def delete(self, scope: MemoryScope, key: str) -> None: ...

    assert isinstance(LegacyStore(), MemoryStore)


# ---------- FakeMemoryStore ----------


@pytest.mark.asyncio
async def test_fake_store_round_trips_an_explicit_label():
    store = FakeMemoryStore()
    await store.save(SCOPE, "k", _value(), provenance=Provenance.UNTRUSTED)
    got = await store.recall(SCOPE, "k")
    assert got is not None
    assert got.provenance is Provenance.UNTRUSTED


@pytest.mark.asyncio
async def test_fake_store_omitting_the_argument_keeps_the_values_own_label():
    """The verifiable-default case: nothing here passes `provenance=`."""
    store = FakeMemoryStore()
    await store.save(SCOPE, "k", _value(provenance=Provenance.UNTRUSTED))
    got = await store.recall(SCOPE, "k")
    assert got is not None
    assert got.provenance is Provenance.UNTRUSTED


@pytest.mark.asyncio
async def test_fake_store_search_carries_provenance_on_the_hit():
    """`search` returns whole values, so the second read path must not lose it."""
    store = FakeMemoryStore()
    await store.save(SCOPE, "k", _value("oat milk"), provenance=Provenance.UNTRUSTED)
    hits = await store.search(SCOPE, "oat")
    assert len(hits) == 1
    assert hits[0].value.provenance is Provenance.UNTRUSTED


# ---------- RedisMemoryStore ----------
#
# Redis integration tests need Docker. These drive the same code against a
# double whose every call is bound against the real `redis.asyncio.Redis`
# signature first, so a call the genuine client would reject fails here too.


class _StrictFakeRedis:
    """In-memory stand-in that rejects any call redis-py itself would reject."""

    def __init__(self) -> None:
        self.kv: dict[Any, Any] = {}
        self.sets: dict[Any, set[Any]] = {}

    def _bind(self, name: str, *args: Any, **kwargs: Any) -> None:
        inspect.signature(getattr(Redis, name)).bind(self, *args, **kwargs)

    async def set(self, *args: Any, **kwargs: Any) -> None:
        self._bind("set", *args, **kwargs)
        self.kv[args[0]] = args[1]

    async def get(self, *args: Any, **kwargs: Any) -> Any:
        self._bind("get", *args, **kwargs)
        return self.kv.get(args[0])

    async def sadd(self, *args: Any, **kwargs: Any) -> int:
        self._bind("sadd", *args, **kwargs)
        self.sets.setdefault(args[0], set()).update(args[1:])
        return len(args) - 1

    # Quoted: this class defines a method named `set`, which shadows the
    # builtin in the class body where annotations are evaluated.
    async def smembers(self, *args: Any, **kwargs: Any) -> "set[Any]":
        self._bind("smembers", *args, **kwargs)
        return set(self.sets.get(args[0], set()))

    async def delete(self, *args: Any, **kwargs: Any) -> int:
        self._bind("delete", *args, **kwargs)
        return sum(self.kv.pop(n, None) is not None for n in args)

    async def srem(self, *args: Any, **kwargs: Any) -> int:
        self._bind("srem", *args, **kwargs)
        self.sets.get(args[0], set()).difference_update(args[1:])
        return 1


class _FakeClient:
    def __init__(self) -> None:
        self.redis = _StrictFakeRedis()
        self.keys = KeyBuilder(prefix="aktest")


def _redis_store() -> tuple[RedisMemoryStore, _StrictFakeRedis]:
    client = _FakeClient()
    return RedisMemoryStore(cast("RedisClient", client)), client.redis


@pytest.mark.asyncio
async def test_redis_store_writes_the_label_into_the_serialised_payload():
    """Not beside it: `recall` rebuilds with model_validate, so anything
    outside the JSON is gone by the time a later turn reads it."""
    store, fake = _redis_store()
    await store.save(SCOPE, "k", _value(), provenance=Provenance.UNTRUSTED)

    (raw,) = fake.kv.values()
    assert json.loads(raw)["provenance"] == "untrusted"


@pytest.mark.asyncio
async def test_redis_store_round_trips_an_explicit_label():
    store, _ = _redis_store()
    await store.save(SCOPE, "k", _value(), provenance=Provenance.UNTRUSTED)
    got = await store.recall(SCOPE, "k")
    assert got is not None
    assert got.provenance is Provenance.UNTRUSTED


@pytest.mark.asyncio
async def test_redis_store_omitting_the_argument_keeps_the_values_own_label():
    store, _ = _redis_store()
    await store.save(SCOPE, "k", _value(provenance=Provenance.UNTRUSTED))
    got = await store.recall(SCOPE, "k")
    assert got is not None
    assert got.provenance is Provenance.UNTRUSTED


@pytest.mark.asyncio
async def test_redis_store_search_carries_provenance_on_the_hit():
    store, _ = _redis_store()
    await store.save(SCOPE, "k", _value("oat milk"), provenance=Provenance.UNTRUSTED)
    hits = await store.search(SCOPE, "oat")
    assert len(hits) == 1
    assert hits[0].value.provenance is Provenance.UNTRUSTED


@pytest.mark.asyncio
async def test_redis_store_reads_a_row_written_before_the_field_existed():
    """The stored bytes of an existing deployment, byte for byte."""
    store, fake = _redis_store()
    key = KeyBuilder(prefix="aktest").memory(SCOPE, "k")
    fake.kv[key] = json.dumps(
        {
            "_v": 1,
            "text": "written last year",
            "payload": {},
            "tags": [],
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-01T00:00:00Z",
        }
    ).encode()

    got = await store.recall(SCOPE, "k")
    assert got is not None
    assert got.provenance is Provenance.SYSTEM
