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

_MISSING = object()


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

    def _id_alias(self, verb: str) -> str | None:
        """The LLM-facing id alias for this resource, read off the verb's spec."""
        name = f"{self._resource}.{verb}"
        if not self._reg.has(name):
            return None
        idp = self._reg.get(name).params.get("id")
        return idp.alias if idp else None

    def _resolve_id(self, verb: str, id: Any, kwargs: dict[str, Any]) -> Any:
        if id is not _MISSING:
            return id
        alias = self._id_alias(verb)
        if alias and alias in kwargs:
            return kwargs.pop(alias)
        raise TypeError(f"{self._resource}.{verb}() requires 'id'")

    def __getattr__(self, verb: str) -> Callable[..., Awaitable[Any]]:
        """Resolve a declared, non-CRUD verb (``kb.cite``, ``graph.link``).

        Only fires when normal attribute lookup misses, so it never shadows the
        fixed CRUD methods below. Underscore names and verbs without a
        registered ``<resource>.<verb>`` OpSpec raise AttributeError, exactly as
        a missing attribute would. Declared writes go through the recorder but
        skip the patch/create field whitelist — they carry structured payloads
        and define their own apply signature.
        """
        if verb.startswith("_"):
            raise AttributeError(verb)
        name = f"{self._resource}.{verb}"
        if not self._reg.has(name):
            raise AttributeError(verb)
        spec = self._reg.get(name)
        param_names = list(spec.params.keys())

        async def call(*args: Any, **kwargs: Any) -> Any:
            for pname, val in zip(param_names, args, strict=False):
                if pname in kwargs:
                    raise TypeError(f"{name}() got multiple values for argument '{pname}'")
                kwargs[pname] = val
            if spec.is_read:
                return await spec.apply(self._ctx, **kwargs)
            return await self._invoke_declared(spec, kwargs)

        return call

    async def _invoke_declared(self, spec: OpSpec, kwargs: dict[str, Any]) -> Any:
        self._charge()
        before = await spec.snapshot(self._ctx, **kwargs) if spec.snapshot else None
        result: Any = await spec.apply(self._ctx, **kwargs)
        after: dict[str, Any] | None = result if isinstance(result, dict) else None  # type: ignore[reportUnknownVariableType]
        inverse = spec.inverse(kwargs, before, after) if spec.inverse else None
        await self._recorder(spec, kwargs, before, after, inverse)
        return result  # type: ignore[reportUnknownVariableType]

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

    async def get(self, id: Any = _MISSING, **kwargs: Any) -> Any:
        return await self._read("get", id=self._resolve_id("get", id, kwargs))

    async def patch(self, id: Any = _MISSING, **fields: Any) -> Any:
        # Resolve the id alias out of fields before the patchable whitelist runs,
        # so an aliased id is never mistaken for a field.
        rid = self._resolve_id("patch", id, fields)
        return await self._write("patch", {"id": rid}, fields)

    async def replace(self, id: Any, **fields: Any) -> Any:
        # In-place edit verb (e.g. find-and-replace) — same write plumbing as
        # patch but a distinct OpSpec, so the consumer can give it its own
        # apply/classify (a body text edit is reversible, not a field set).
        return await self._write("replace", {"id": id}, fields)

    async def create(self, **fields: Any) -> Any:
        return await self._write("create", {}, fields)

    async def delete(self, id: Any = _MISSING, **kwargs: Any) -> Any:
        rid = self._resolve_id("delete", id, kwargs)
        spec = self._spec("delete")
        self._charge()
        before = await spec.snapshot(self._ctx, id=rid) if spec.snapshot else None
        result: Any = await spec.apply(self._ctx, id=rid)
        after: dict[str, Any] | None = result if isinstance(result, dict) else None  # type: ignore[reportUnknownVariableType]
        inverse = spec.inverse({"id": rid}, before, after) if spec.inverse else None
        await self._recorder(spec, {"id": rid}, before, after, inverse)
        return result  # type: ignore[reportUnknownVariableType]
