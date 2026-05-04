"""Tool-result caching keyed by ``(name, hash(arguments))``.

Uses any ``CheckpointStore``-compatible bytes-KV. Cache hits flip ``cached=True``
on the returned result.
"""

import hashlib
import json
from typing import Any

from agentkit._ids import CheckpointId
from agentkit.store.checkpoint import CheckpointStore
from agentkit.tools.spec import ToolResult


def cache_key(tool_name: str, arguments: dict[str, Any]) -> CheckpointId:
    """Compute a deterministic cache key.

    Uses sorted keys so argument-order doesn't change the hash. Returns a
    CheckpointId so any CheckpointStore can serve as the cache backend.
    """
    canonical = json.dumps(arguments, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256(f"{tool_name}|{canonical}".encode()).hexdigest()
    return CheckpointId(f"toolcache:{digest}")


class ToolResultCache:
    def __init__(self, store: CheckpointStore) -> None:
        self._store = store

    async def load(self, key: CheckpointId) -> ToolResult | None:
        raw = await self._store.load(key)
        if raw is None:
            return None
        result = ToolResult.model_validate_json(raw)
        return result.model_copy(update={"cached": True})

    async def store(self, key: CheckpointId, result: ToolResult, *, ttl_seconds: int) -> None:
        # We pass through to a CheckpointStore which has its own TTL; honouring
        # ttl_seconds here is per-backend (RedisCheckpointStore takes ttl in __init__).
        # For tool cache, recommend wiring a dedicated CheckpointStore with appropriate TTL
        # OR provide a TTL-aware variant in v0.2.
        await self._store.save(key, result.model_dump_json().encode("utf-8"))
