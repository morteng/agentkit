"""Encode canonical agentkit tool names for OpenAI-compatible wire APIs.

Agentkit uses qualified names (for example ``acme.search``) internally.
OpenAI-compatible function schemas impose a stricter identifier grammar and do
not accept dots.  This module keeps the canonical names unchanged in history,
routing, and tool events while translating names only at the provider boundary.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from agentkit._content import ToolUseBlock
from agentkit.providers.base import NamedToolChoice

if TYPE_CHECKING:
    from collections.abc import Iterable

    from agentkit.providers.base import ProviderRequest

WIRE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
MAX_WIRE_NAME_LENGTH = 64
UNKNOWN_INVALID_TOOL_NAME = "__invalid_tool_name__"


def is_wire_safe_name(name: str) -> bool:
    """Return whether *name* satisfies the OpenAI function-name grammar."""

    return (
        bool(name) and len(name) <= MAX_WIRE_NAME_LENGTH and bool(WIRE_NAME_PATTERN.fullmatch(name))
    )


@dataclass(frozen=True)
class ToolNameCodec:
    """Request-scoped, bidirectional canonical/wire tool-name mapping.

    Canonical names are the names used by the agentkit registry and session
    history.  Wire names are emitted to, and accepted from, an
    OpenAI-compatible provider.  The mapping is deliberately scoped to one
    request so aliases can be collision-free without changing persisted state.
    """

    canonical_to_wire: dict[str, str]
    wire_to_canonical: dict[str, str]

    @classmethod
    def from_names(cls, names: Iterable[str]) -> ToolNameCodec:
        """Build a deterministic codec for the supplied canonical names."""

        canonical_names = sorted(set(names))
        if any(not name for name in canonical_names):
            raise ValueError("tool names must be non-empty strings")

        canonical_to_wire: dict[str, str] = {}
        wire_to_canonical: dict[str, str] = {}
        used_wire_names: set[str] = set()

        # The natural ``__`` encodings of the non-wire-safe names in this set.
        # A wire-safe canonical name that *equals* one of these is ambiguous:
        # it may be a genuine flat tool, or it may be a dotted tool's own wire
        # alias that a decode miss on an earlier request preserved and the
        # session then stored into history as if it were canonical.  The dotted
        # original must win that collision.  When the echoed alias won (the
        # previous behaviour), the poisoned history entry held the natural
        # alias forever: the real tool was advertised under a ``_2`` suffix,
        # every call the model made to the name it had seen work decoded to the
        # poisoned entry, and the registry answered "unknown tool" for a tool
        # it had — a self-perpetuating failure observed live
        # (``torrent__torrent_add`` failing minutes after ``torrent.torrent_add``
        # ran twice).  With the dotted original winning, the same call decodes
        # back to the registered name and the session heals itself.
        encoded_candidates = {
            _candidate_wire_name(name) for name in canonical_names if not is_wire_safe_name(name)
        }

        # Preserve already-safe names whenever possible.  This keeps ordinary
        # flat tools byte-compatible and leaves only qualified/invalid names
        # translated.  Reserve these first so an encoded name can never steal
        # an exact alias from another canonical tool — except when the safe
        # name IS another tool's natural encoding (see above): that name goes
        # through the suffix loop instead, and sorted order guarantees the
        # dotted original claims the alias first (``.`` sorts before ``_``).
        for name in canonical_names:
            if is_wire_safe_name(name) and name not in encoded_candidates:
                canonical_to_wire[name] = name
                wire_to_canonical[name] = name
                used_wire_names.add(name)

        for name in canonical_names:
            if name in canonical_to_wire:
                continue
            candidate = _candidate_wire_name(name)
            alias = candidate
            suffix = 2
            while alias in used_wire_names:
                alias = _with_suffix(candidate, suffix)
                suffix += 1
            if not is_wire_safe_name(alias):  # pragma: no cover - defensive
                raise ValueError(f"could not encode tool name safely: {name!r}")
            canonical_to_wire[name] = alias
            wire_to_canonical[alias] = name
            used_wire_names.add(alias)

        if len(canonical_to_wire) != len(wire_to_canonical):  # pragma: no cover
            raise ValueError("tool-name codec produced a non-bijective mapping")
        return cls(canonical_to_wire=canonical_to_wire, wire_to_canonical=wire_to_canonical)

    @classmethod
    def from_request(cls, request: ProviderRequest) -> ToolNameCodec:
        """Build a codec covering tools, named choices, and replayed history."""

        names: list[str] = [tool.name for tool in request.tools]
        if isinstance(request.tool_choice, NamedToolChoice):
            names.append(request.tool_choice.name)

        for message in request.messages:
            names.extend(block.name for block in message.content if isinstance(block, ToolUseBlock))
        return cls.from_names(names)

    def encode(self, canonical_name: str) -> str:
        """Return the provider-safe alias for a canonical name."""

        try:
            return self.canonical_to_wire[canonical_name]
        except KeyError as exc:
            raise ValueError(
                f"tool name was not included in this request: {canonical_name!r}"
            ) from exc

    def decode(self, wire_name: str) -> str:
        """Decode a provider alias, failing closed for malformed names.

        A name that is not a known alias but IS a known canonical name is
        returned as canonical: models echo the dotted names they see in
        history text and error diagnostics ("Did you mean acme.search?"), and
        such a name identifies exactly one tool of this request — treating it
        as malformed would turn correct intent into the sentinel.

        A provider may return a safe-but-unknown name; preserving it lets the
        existing unknown-tool path produce its normal diagnostic.  Malformed
        names are replaced with a safe sentinel so they cannot poison session
        history and break the next provider request.
        """

        canonical_name = self.wire_to_canonical.get(wire_name)
        if canonical_name is not None:
            return canonical_name
        if wire_name in self.canonical_to_wire:
            # Only reachable for canonical names that are not their own alias
            # (dotted names, or a flat name displaced from its alias by a
            # dotted sibling's encoding — the alias lookup above wins for
            # every identity-mapped name).
            return wire_name
        return wire_name if is_wire_safe_name(wire_name) else UNKNOWN_INVALID_TOOL_NAME


def _candidate_wire_name(name: str) -> str:
    """Create a readable safe candidate, with a bounded hash fallback."""

    dotted = name.replace(".", "__")
    if is_wire_safe_name(dotted):
        return dotted
    # The digest is bounded and uses only hexadecimal characters.  Collisions
    # are handled by the deterministic suffix loop in ``from_names``.
    return f"ak_{hashlib.sha256(name.encode('utf-8')).hexdigest()[:24]}"


def _with_suffix(candidate: str, suffix: int) -> str:
    suffix_text = f"_{suffix}"
    return f"{candidate[: MAX_WIRE_NAME_LENGTH - len(suffix_text)]}{suffix_text}"


__all__ = [
    "MAX_WIRE_NAME_LENGTH",
    "UNKNOWN_INVALID_TOOL_NAME",
    "WIRE_NAME_PATTERN",
    "ToolNameCodec",
    "is_wire_safe_name",
]
