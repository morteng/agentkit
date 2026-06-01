"""ResourceNamespace — the uniform search/get/patch/create/delete surface.

Generic in shape, typed in content: each verb looks up `<resource>.<verb>` in
the registry. Writes validate field kwargs against the spec's `patchable`
whitelist, call `spec.apply`, then record via the injected recorder using the
spec's snapshot/inverse. Reads call apply and return.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from agentkit.resources.types import OpSpec

if TYPE_CHECKING:
    from agentkit.resources.registry import OpRegistry

Recorder = Callable[
    [
        OpSpec,
        dict[str, Any],
        dict[str, Any] | None,
        dict[str, Any] | None,
        dict[str, Any] | None,
    ],
    Awaitable[None],
]


class ResourceNamespace:
    def __init__(
        self,
        resource: str,
        registry: OpRegistry,
        *,
        ctx: Any,
        recorder: Recorder,
        op_charge: Callable[[], None],
    ) -> None:
        self._resource = resource
        self._reg = registry
        self._ctx = ctx
        self._recorder = recorder
        self._charge = op_charge

    def _spec(self, verb: str) -> OpSpec:
        return self._reg.get(f"{self._resource}.{verb}")

    async def _read(self, verb: str, **kwargs: Any) -> Any:
        spec = self._spec(verb)
        return await spec.apply(self._ctx, **kwargs)

    async def _write(self, verb: str, id_kwargs: dict[str, Any], fields: dict[str, Any]) -> Any:
        spec = self._spec(verb)
        # Field whitelist: only fields-bearing verbs (patch/create) enforce it.
        if spec.patchable or fields:
            bad = set(fields) - set(spec.patchable)
            if bad:
                raise ValueError(f"fields not patchable on {self._resource}: {sorted(bad)}")
        self._charge()
        kwargs: dict[str, Any] = {**id_kwargs, **fields}
        before = await spec.snapshot(self._ctx, **kwargs) if spec.snapshot else None
        result: Any = await spec.apply(self._ctx, **kwargs)
        after: dict[str, Any] | None = result if isinstance(result, dict) else None  # type: ignore[reportUnknownVariableType]
        inverse = spec.inverse(kwargs, before, after) if spec.inverse else None
        await self._recorder(spec, kwargs, before, after, inverse)
        return result  # type: ignore[reportUnknownVariableType]

    # Uniform verbs ---------------------------------------------------------
    async def search(self, query: str = "", **filters: Any) -> Any:
        return await self._read("search", query=query, **filters)

    async def get(self, id: Any) -> Any:
        return await self._read("get", id=id)

    async def patch(self, id: Any, **fields: Any) -> Any:
        return await self._write("patch", {"id": id}, fields)

    async def create(self, **fields: Any) -> Any:
        return await self._write("create", {}, fields)

    async def delete(self, id: Any) -> Any:
        spec = self._spec("delete")
        self._charge()
        before = await spec.snapshot(self._ctx, id=id) if spec.snapshot else None
        result: Any = await spec.apply(self._ctx, id=id)
        after: dict[str, Any] | None = result if isinstance(result, dict) else None  # type: ignore[reportUnknownVariableType]
        inverse = spec.inverse({"id": id}, before, after) if spec.inverse else None
        await self._recorder(spec, {"id": id}, before, after, inverse)
        return result  # type: ignore[reportUnknownVariableType]
