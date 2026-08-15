"""Centralised Redis key naming. All keys go through KeyBuilder.

Convention: ``{prefix}:{kind}:{id}[:{sub}]``
- prefix: deployment-level namespace (default "agentkit")
- kind:   resource type ("sess", "msgs", "mem", "ckpt", "owner")
- id:     resource identifier

Every id, scope component and memory key is percent-encoded before it is
interpolated. Redis has no key hierarchy — ``:`` is a convention, not a
delimiter it enforces — so an unescaped ``:`` inside a *value* silently
becomes a structural separator. A model-controlled memory key
``"u:1:secret"`` would otherwise land in a different scope's namespace than
the one the caller asked for, and a key of ``"_index"`` would clobber that
scope's key index outright. Escaping restores the invariant the layout
assumes: one value, one segment.

Only ``prefix`` is left verbatim — it is deployment configuration, chosen by
the operator, never derived from a request.
"""

from urllib.parse import quote

from agentkit._ids import CheckpointId, OwnerId, SessionId
from agentkit.store.memory import MemoryScope

_MEMORY_INDEX_SUFFIX = "%index"
"""Reserved final segment of a scope's key-index SET.

Percent-encoding escapes ``%`` itself (to ``%25``), so no encoded user key can
ever start with a bare ``%`` — the index key is unreachable from the key
namespace by construction, rather than by hoping nobody stores ``"_index"``.
"""

MAX_MEMORY_KEY_LENGTH = 512
"""Longest accepted memory key, before escaping.

Not a Redis limit (keys may be up to 512 MB); a sanity bound so a runaway
model cannot turn one memory write into a multi-megabyte key.
"""


def escape_key_part(value: str) -> str:
    """Percent-encode one key segment so it cannot introduce structure.

    ``quote(safe="")`` escapes ``:``, ``%`` and every other reserved character,
    leaving only the RFC 3986 unreserved set. The mapping is injective (``%``
    itself is escaped), so distinct inputs stay distinct keys — which is what
    makes cross-scope collisions impossible rather than merely unlikely.
    """
    return quote(value, safe="")


def validate_memory_key(key: str) -> None:
    """Reject memory keys that should never reach the store.

    Escaping already makes a hostile key *harmless*; this is the separate
    question of whether it is *sensible*. Callers on the write path (the memory
    store, the ``memory`` builtin tool) should call this so a bad key fails at
    the boundary with a clear message instead of becoming an unreachable row.

    Raises:
        ValueError: the key is empty, whitespace-only, or over
            :data:`MAX_MEMORY_KEY_LENGTH` characters.
    """
    if not key or not key.strip():
        raise ValueError("memory key must be a non-empty, non-whitespace string")
    if len(key) > MAX_MEMORY_KEY_LENGTH:
        raise ValueError(
            f"memory key too long: {len(key)} characters (max {MAX_MEMORY_KEY_LENGTH})"
        )


class KeyBuilder:
    def __init__(self, prefix: str = "agentkit") -> None:
        self._prefix = prefix

    def session(self, sid: SessionId) -> str:
        return f"{self._prefix}:sess:{escape_key_part(sid)}"

    def messages(self, sid: SessionId) -> str:
        return f"{self._prefix}:msgs:{escape_key_part(sid)}"

    def owner_index(self, owner: OwnerId) -> str:
        return f"{self._prefix}:owner:{escape_key_part(owner)}:sessions"

    def memory(self, scope: MemoryScope, key: str) -> str:
        return f"{self._prefix}:mem:{self._scope_part(scope)}:{escape_key_part(key)}"

    def memory_index(self, scope: MemoryScope) -> str:
        return f"{self._prefix}:mem:{self._scope_part(scope)}:{_MEMORY_INDEX_SUFFIX}"

    def checkpoint(self, cid: CheckpointId) -> str:
        return f"{self._prefix}:ckpt:{escape_key_part(cid)}"

    def event_channel(self, sid: SessionId) -> str:
        return f"{self._prefix}:events:{escape_key_part(sid)}"

    def event_buffer(self, sid: SessionId) -> str:
        return f"{self._prefix}:event-buffer:{escape_key_part(sid)}"

    @staticmethod
    def _scope_part(scope: MemoryScope) -> str:
        """Render a scope as ``ns[:t:tenant][:u:user][:s:session]``.

        The single-letter tags sit at fixed positions and every value is
        escaped, so the rendering is injective: two different scopes cannot
        produce the same string, and no value can impersonate a tag.
        """
        parts = [escape_key_part(scope.namespace)]
        if scope.tenant_id:
            parts.append(f"t:{escape_key_part(scope.tenant_id)}")
        if scope.user_id:
            parts.append(f"u:{escape_key_part(scope.user_id)}")
        if scope.session_id:
            parts.append(f"s:{escape_key_part(scope.session_id)}")
        return ":".join(parts)
