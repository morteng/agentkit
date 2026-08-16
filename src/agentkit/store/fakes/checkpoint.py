"""In-memory FakeCheckpointStore."""

from agentkit._ids import CheckpointId
from agentkit.store.checkpoint import CheckpointPayload, CheckpointStore


class FakeCheckpointStore(CheckpointStore):
    def __init__(self) -> None:
        self._data: dict[CheckpointId, CheckpointPayload] = {}

    async def save(self, checkpoint_id: CheckpointId, payload: CheckpointPayload) -> None:
        self._data[checkpoint_id] = payload

    async def load(self, checkpoint_id: CheckpointId) -> CheckpointPayload | None:
        return self._data.get(checkpoint_id)

    async def delete(self, checkpoint_id: CheckpointId) -> None:
        self._data.pop(checkpoint_id, None)

    async def list_ids(self, prefix: str = "") -> list[CheckpointId]:
        # Sorted, where the Redis backend cannot promise any order. A fake that
        # returned a *stable arbitrary* order (dict insertion) would let a test
        # accidentally depend on ordering the real store does not provide.
        return sorted(cid for cid in self._data if cid.startswith(prefix))
