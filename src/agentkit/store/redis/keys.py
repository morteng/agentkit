"""Centralised Redis key naming. All keys go through KeyBuilder.

Convention: ``{prefix}:{kind}:{id}[:{sub}]``
- prefix: deployment-level namespace (default "agentkit")
- kind:   resource type ("sess", "msgs", "mem", "ckpt", "owner")
- id:     resource identifier
"""

from agentkit._ids import CheckpointId, OwnerId, SessionId
from agentkit.store.memory import MemoryScope


class KeyBuilder:
    def __init__(self, prefix: str = "agentkit") -> None:
        self._prefix = prefix

    def session(self, sid: SessionId) -> str:
        return f"{self._prefix}:sess:{sid}"

    def messages(self, sid: SessionId) -> str:
        return f"{self._prefix}:msgs:{sid}"

    def owner_index(self, owner: OwnerId) -> str:
        return f"{self._prefix}:owner:{owner}:sessions"

    def memory(self, scope: MemoryScope, key: str) -> str:
        return f"{self._prefix}:mem:{self._scope_part(scope)}:{key}"

    def memory_index(self, scope: MemoryScope) -> str:
        return f"{self._prefix}:mem:{self._scope_part(scope)}:_index"

    def checkpoint(self, cid: CheckpointId) -> str:
        return f"{self._prefix}:ckpt:{cid}"

    def event_channel(self, sid: SessionId) -> str:
        return f"{self._prefix}:events:{sid}"

    def event_buffer(self, sid: SessionId) -> str:
        return f"{self._prefix}:event-buffer:{sid}"

    @staticmethod
    def _scope_part(scope: MemoryScope) -> str:
        parts = [scope.namespace]
        if scope.tenant_id:
            parts.append(f"t:{scope.tenant_id}")
        if scope.user_id:
            parts.append(f"u:{scope.user_id}")
        if scope.session_id:
            parts.append(f"s:{scope.session_id}")
        return ":".join(parts)
