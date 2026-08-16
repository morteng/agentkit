"""Checkpoint store protocol — resumable mid-turn state."""

from typing import Protocol, runtime_checkable

from agentkit._ids import CheckpointId, TurnId

# CheckpointPayload kept opaque (`bytes`) at the protocol level so the
# orchestrator can choose its own serialisation. Redis backend stores raw bytes.
type CheckpointPayload = bytes

APPROVAL_CHECKPOINT_PREFIX = "approval:"
"""Namespace for the checkpoint an approval suspend writes.

A checkpoint id is otherwise opaque, and the store is shared with anything else
that wants to persist mid-turn state. The prefix is what lets
:meth:`agentkit.session.AgentSession.expire_due` enumerate *only* the pending
approvals instead of loading and JSON-parsing every checkpoint in the
deployment to find out which ones are approvals. It is a naming convention, not
an enforced schema, so it lives here next to the two functions that construct
and destructure it rather than being re-spelled at each call site — the four
hand-written f-strings it replaces were one typo away from a sweep that
silently matched nothing.
"""


def approval_checkpoint_id(turn_id: TurnId) -> CheckpointId:
    """The checkpoint id an approval suspend on ``turn_id`` is stored under."""
    return CheckpointId(f"{APPROVAL_CHECKPOINT_PREFIX}{turn_id}")


def turn_id_from_approval_checkpoint(checkpoint_id: CheckpointId) -> TurnId | None:
    """The turn behind an approval checkpoint id, or ``None`` if it is not one.

    The inverse of :func:`approval_checkpoint_id`. Returns ``None`` rather than
    raising because its caller is a sweep over whatever ids the store happens
    to hold: a checkpoint written by some other part of the host application is
    not an error, it is simply not this function's business.
    """
    if not checkpoint_id.startswith(APPROVAL_CHECKPOINT_PREFIX):
        return None
    return TurnId(checkpoint_id[len(APPROVAL_CHECKPOINT_PREFIX) :])


@runtime_checkable
class CheckpointStore(Protocol):
    async def save(self, checkpoint_id: CheckpointId, payload: CheckpointPayload) -> None: ...
    async def load(self, checkpoint_id: CheckpointId) -> CheckpointPayload | None: ...
    async def delete(self, checkpoint_id: CheckpointId) -> None: ...


@runtime_checkable
class EnumerableCheckpointStore(CheckpointStore, Protocol):
    """A :class:`CheckpointStore` that can also list what it holds.

    Kept as a separate protocol rather than a fourth method on
    :class:`CheckpointStore` because ``CheckpointStore`` is a published
    interface that host applications implement against their own storage.
    Adding a required method would break every one of those on upgrade, to buy
    a capability most of them do not need — a store used only for
    suspend/resume never has to enumerate anything.

    Callers therefore feature-detect with
    ``isinstance(store, EnumerableCheckpointStore)`` (structural, via
    ``runtime_checkable``, so an implementation does not need to import or
    inherit this) and are expected to fail loudly rather than quietly do
    nothing when the check fails. A sweep that cannot see the keyspace and
    reports "nothing was due" is the same silent-consent failure the sweep
    exists to remove.
    """

    async def list_ids(self, prefix: str = "") -> list[CheckpointId]:
        """Every stored checkpoint id starting with ``prefix``.

        Order is unspecified — callers that need determinism sort. An id whose
        underlying entry expires between this listing and a subsequent
        :meth:`load` is a normal race, not an error: a caller must tolerate
        ``load`` returning ``None`` for an id this returned.
        """
        ...
