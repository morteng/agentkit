"""RedisCheckpointStore.list_ids — the key round-trip, without a Redis.

``list_ids`` is the only place in the codebase that has to *reverse* the
KeyBuilder: every other read is given the id it wants and escapes forward. The
inverse is where a bug hides, because both halves look right in isolation —
``escape_key_part`` percent-encodes ``:`` to ``%3A``, so an approval prefix
that is not escaped on its way into the SCAN glob matches nothing at all, and a
key that is not unquoted on its way out yields a CheckpointId of
``"approval%3A01M0…"`` that no subsequent ``load`` can find. Both failures are
silent: the sweep simply reports that nothing was due.

The integration suite covers this against a real Redis
(tests/integration/store/test_redis_checkpoint.py), but that needs Docker. This
runs everywhere, so the escaping cannot regress unnoticed on a machine without
a daemon.

The double is deliberately strict — every call it receives is bound against the
genuine ``redis.asyncio.Redis`` signature, so a call the real client would
reject fails here too — and it stores against the *real* KeyBuilder output, so
the test exercises the actual key layout rather than a restatement of it.
"""

import fnmatch
import inspect
from collections.abc import AsyncIterator
from typing import Any

import pytest
from redis.asyncio import Redis

from agentkit._ids import CheckpointId
from agentkit.store.redis.checkpoint import RedisCheckpointStore
from agentkit.store.redis.keys import KeyBuilder


class _StrictFakeRedis:
    """The three Redis calls RedisCheckpointStore makes, over a dict.

    Each one binds its arguments against the real method's signature first.
    ``scan_iter`` in particular is easy to call with a keyword the installed
    client does not have (``match`` vs ``pattern`` has moved before); binding
    means such a call fails here instead of passing against a permissive
    ``**kwargs`` stub and then failing in production.
    """

    def __init__(self) -> None:
        self.data: dict[str, bytes] = {}

    @staticmethod
    def _bind(name: str, *args: Any, **kwargs: Any) -> None:
        """Bind against the real method, and refuse the ``**kwargs`` escape hatch.

        The stub's own narrow signature already rejects a keyword neither it
        nor the client accepts. This adds the case that signature cannot see:
        ``Redis.scan_iter`` ends in ``**kwargs``, so a plain ``bind`` waves
        through any keyword at all, and a double built on ``**kwargs`` would be
        exactly as permissive as the stub it is supposed to replace. Anything
        landing in the VAR_KEYWORD bucket is a name the client does not really
        have, so it fails here rather than in production.
        """
        signature = inspect.signature(getattr(Redis, name))
        bound = signature.bind(None, *args, **kwargs)
        for parameter in signature.parameters.values():
            if parameter.kind is inspect.Parameter.VAR_KEYWORD:
                extra = bound.arguments.get(parameter.name) or {}
                if extra:
                    raise TypeError(
                        f"Redis.{name}() got unexpected keyword(s) {sorted(extra)} — "
                        f"they only bind because the real signature ends in **kwargs"
                    )

    async def set(self, key: str, value: bytes, ex: int | None = None) -> None:
        self._bind("set", key, value, ex=ex)
        self.data[key] = value

    async def get(self, key: str) -> bytes | None:
        self._bind("get", key)
        return self.data.get(key)

    async def delete(self, key: str) -> None:
        self._bind("delete", key)
        self.data.pop(key, None)

    def scan_iter(self, match: str | None = None, count: int | None = None) -> AsyncIterator[bytes]:
        self._bind("scan_iter", match=match, count=count)

        async def _iter() -> AsyncIterator[bytes]:
            # Reverse-sorted so the test cannot pass by accident on insertion
            # order: the real SCAN promises no order at all.
            for key in sorted(self.data, reverse=True):
                if match is None or fnmatch.fnmatchcase(key, match):
                    yield key.encode()

        return _iter()


class _FakeClient:
    def __init__(self, prefix: str = "agentkit") -> None:
        self.redis = _StrictFakeRedis()
        self.keys = KeyBuilder(prefix=prefix)


def _store(prefix: str = "agentkit") -> tuple[RedisCheckpointStore, _FakeClient]:
    client = _FakeClient(prefix)
    return RedisCheckpointStore(client), client  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_list_ids_round_trips_ids_through_the_escaped_key_layout():
    """FAILS PRE-FIX (RedisCheckpointStore has no list_ids).

    A checkpoint id contains a ``:``, which the KeyBuilder escapes. The id that
    comes back out must be the id that went in — byte for byte, usable as an
    argument to ``load``.
    """
    store, _ = _store()
    ids = [CheckpointId("approval:01M0AAA"), CheckpointId("approval:01M0BBB")]
    for cid in ids:
        await store.save(cid, b"{}")

    listed = await store.list_ids("approval:")

    assert sorted(listed) == sorted(ids)
    # Not merely equal-looking: each one still addresses its own payload.
    for cid in listed:
        assert await store.load(cid) == b"{}"


@pytest.mark.asyncio
async def test_list_ids_filters_out_other_subsystems_checkpoints():
    store, _ = _store()
    await store.save(CheckpointId("approval:01M0AAA"), b"{}")
    await store.save(CheckpointId("some-other-subsystem:42"), b"{}")

    assert await store.list_ids("approval:") == [CheckpointId("approval:01M0AAA")]


@pytest.mark.asyncio
async def test_list_ids_does_not_leak_across_deployment_prefixes():
    """Two agentkit deployments sharing one Redis must not sweep each other's
    approvals. The glob is anchored on the KeyBuilder prefix, not just on
    ``approval:``."""
    mine, mine_client = _store(prefix="mine")
    theirs, _ = _store(prefix="theirs")
    await mine.save(CheckpointId("approval:01M0AAA"), b"{}")
    # Same underlying dict is not shared, so plant the foreign key by hand
    # exactly as the other deployment's KeyBuilder would have written it.
    foreign_key = KeyBuilder(prefix="theirs").checkpoint(CheckpointId("approval:01M0BBB"))
    mine_client.redis.data[foreign_key] = b"{}"

    assert await mine.list_ids("approval:") == [CheckpointId("approval:01M0AAA")]
    assert await theirs.list_ids("approval:") == []


@pytest.mark.asyncio
async def test_a_prefix_containing_glob_metacharacters_cannot_widen_the_match():
    """Percent-encoding leaves only the RFC 3986 unreserved set, so ``*`` and
    friends never reach Redis as glob syntax. Without that, a prefix of ``*``
    would match every checkpoint in the deployment — and this sweep deletes
    what it matches."""
    store, _ = _store()
    await store.save(CheckpointId("approval:01M0AAA"), b"{}")

    assert await store.list_ids("*") == []
    assert await store.list_ids("appro*") == []
    assert await store.list_ids("[a-z]pproval:") == []


@pytest.mark.asyncio
async def test_list_ids_with_no_prefix_returns_everything():
    store, _ = _store()
    await store.save(CheckpointId("approval:01M0AAA"), b"{}")
    await store.save(CheckpointId("other:x"), b"{}")

    assert sorted(await store.list_ids()) == ["approval:01M0AAA", "other:x"]
