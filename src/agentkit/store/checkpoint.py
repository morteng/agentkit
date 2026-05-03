"""Checkpoint store protocol — resumable mid-turn state."""

from typing import Protocol, runtime_checkable

from agentkit._ids import CheckpointId

# CheckpointPayload kept opaque (`bytes`) at the protocol level so the
# orchestrator can choose its own serialisation. Redis backend stores raw bytes.
type CheckpointPayload = bytes


@runtime_checkable
class CheckpointStore(Protocol):
    async def save(self, checkpoint_id: CheckpointId, payload: CheckpointPayload) -> None: ...
    async def load(self, checkpoint_id: CheckpointId) -> CheckpointPayload | None: ...
    async def delete(self, checkpoint_id: CheckpointId) -> None: ...
