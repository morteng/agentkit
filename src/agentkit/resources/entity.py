"""EntitySpec — declare a scriptable entity once; ``build_crud_specs`` emits the
uniform get/search/patch/delete OpSpecs wired to shared generic callables.

The consuming app supplies only the per-entity bits: how to view a row, load
one, list many, apply a patch (service-layer), and soft-delete it, plus the
reversibility classifiers. Reads, snapshot, and inverse-op construction are
written here once and shared across every entity. An entity whose pre-mutation
state cannot be reconstructed from its view (e.g. content, whose body lives on a
translation row the view omits) overrides ``snapshot``/``inverse_patch``/
``inverse_delete`` to keep its ledger shapes exact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from agentkit.resources.types import ClassifyFn, InverseFn, OpSpec, Reversibility, SnapshotFn

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


def _default_gated(static_kwargs: dict[str, Any], dynamic_args: frozenset[str]) -> Reversibility:
    """Conservative default: an unclassified mutation forces approval."""
    return Reversibility.GATED


def _to_dict(view: Any) -> dict[str, Any]:
    """Coerce a view (dataclass instance or dict) to a jsonable dict.

    A dataclass instance exposes ``__dict__``; a plain dict has no ``__dict__``
    attribute so ``getattr`` returns it unchanged.
    """
    return getattr(view, "__dict__", view)


@dataclass
class EntitySpec:
    """Declarative description of one entity's scriptable CRUD surface."""

    resource: str  # "locations", "kb", "content"
    subject_type: str  # ledger subject_type
    patchable: frozenset[str]  # fields a patch may set
    view: Callable[[Any], Any]  # row -> jsonable view (dataclass or dict)
    load: Callable[..., Awaitable[Any]]  # async (ctx, id) -> row (eager-loads view deps)
    list_: Callable[..., Awaitable[list[Any]]]  # async (ctx, *, query, **filters) -> rows
    patch_adapter: Callable[..., Awaitable[None]]  # async (ctx, row, fields) -> None
    soft_delete: Callable[..., Awaitable[None]]  # async (ctx, row) -> None
    snapshot_fields: frozenset[str]  # subset of view keys for the generic snapshot
    classify: ClassifyFn = _default_gated  # patch reversibility
    delete_classify: ClassifyFn = _default_gated  # delete reversibility
    # Optional overrides — default to the generic implementations below.
    search_view: Callable[[Any], Any] | None = None  # list-row -> view (defaults to ``view``)
    snapshot: SnapshotFn | None = None  # before_state capture (defaults to view-of-snapshot_fields)
    inverse_patch: InverseFn | None = None  # patch inverse-op (defaults to restore touched fields)
    inverse_delete: InverseFn | None = None  # delete inverse-op (defaults to {resource}.restore)
    delete_action_kind: str = "soft_delete"


def _make_snapshot(spec: EntitySpec) -> SnapshotFn:
    async def snapshot(ctx: Any, *, id: Any, **_: Any) -> dict[str, Any] | None:
        try:
            row = await spec.load(ctx, id)
        except Exception:
            return None
        vd = _to_dict(spec.view(row))
        return {k: vd[k] for k in spec.snapshot_fields if k in vd}

    return snapshot


def _make_inverse_patch(spec: EntitySpec) -> InverseFn:
    def inverse_patch(
        kwargs: dict[str, Any], before: dict[str, Any] | None, after: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        if not before:
            return None
        touched = {k for k in kwargs if k != "id" and k in spec.patchable}
        fields = {k: before[k] for k in touched if k in before}
        if not fields:
            return None
        return {"op": f"{spec.resource}.patch", "id": str(kwargs["id"]), "fields": fields}

    return inverse_patch


def _make_inverse_delete(spec: EntitySpec) -> InverseFn:
    def inverse_delete(
        kwargs: dict[str, Any], before: dict[str, Any] | None, after: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        return {"op": f"{spec.resource}.restore", "id": str(kwargs["id"])}

    return inverse_delete


def build_crud_specs(spec: EntitySpec) -> list[OpSpec]:
    """Emit the four uniform CRUD OpSpecs for one declared entity.

    get/search are reads; patch/delete carry snapshot + inverse + classify so the
    approval scanner grades them and the ledger can reverse them.
    """
    search_view = spec.search_view or spec.view
    snapshot = spec.snapshot or _make_snapshot(spec)
    inverse_patch = spec.inverse_patch or _make_inverse_patch(spec)
    inverse_delete = spec.inverse_delete or _make_inverse_delete(spec)

    async def _get(ctx: Any, *, id: Any) -> dict[str, Any]:
        return _to_dict(spec.view(await spec.load(ctx, id)))

    async def _search(ctx: Any, *, query: str = "", **filters: Any) -> list[dict[str, Any]]:
        rows = await spec.list_(ctx, query=query, **filters)
        return [_to_dict(search_view(r)) for r in rows]

    async def _patch(ctx: Any, *, id: Any, **fields: Any) -> dict[str, Any]:
        row = await spec.load(ctx, id)
        await spec.patch_adapter(ctx, row, fields)
        # patch_adapter writes relationships through the session, so the
        # identity-mapped instance can hold stale state — expire then reload.
        ctx.db.expire(row)
        return _to_dict(spec.view(await spec.load(ctx, id)))

    async def _delete(ctx: Any, *, id: Any) -> dict[str, Any]:
        await spec.soft_delete(ctx, await spec.load(ctx, id))
        return {"id": str(id), "deleted": True}

    return [
        OpSpec(name=f"{spec.resource}.get", apply=_get, is_read=True),
        OpSpec(name=f"{spec.resource}.search", apply=_search, is_read=True),
        OpSpec(
            name=f"{spec.resource}.patch",
            apply=_patch,
            subject_type=spec.subject_type,
            patchable=spec.patchable,
            snapshot=snapshot,
            inverse=inverse_patch,
            classify=spec.classify,
        ),
        OpSpec(
            name=f"{spec.resource}.delete",
            apply=_delete,
            action_kind=spec.delete_action_kind,
            subject_type=spec.subject_type,
            snapshot=snapshot,
            inverse=inverse_delete,
            classify=spec.delete_classify,
        ),
    ]
