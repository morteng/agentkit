"""Strongly-typed identifier wrappers backed by ULIDs.

ULIDs (https://github.com/ulid/spec) are 26-char Crockford-base32 strings,
sortable by creation time. We use NewType to distinguish identifier kinds
at the type level without runtime cost.
"""

from collections.abc import Callable
from typing import NewType

from ulid import ULID

SessionId = NewType("SessionId", str)
TurnId = NewType("TurnId", str)
MessageId = NewType("MessageId", str)
CheckpointId = NewType("CheckpointId", str)
EventId = NewType("EventId", str)
OwnerId = NewType("OwnerId", str)


def new_id[IdT: str](kind: Callable[[str], IdT]) -> IdT:
    """Mint a new ULID-backed identifier of the given kind.

    The return type is the NewType — callers see e.g. ``SessionId`` not ``str``.
    """
    return kind(str(ULID()))
