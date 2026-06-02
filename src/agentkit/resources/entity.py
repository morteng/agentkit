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

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from agentkit.resources.types import ClassifyFn, InverseFn, OpSpec, Reversibility, SnapshotFn

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


def _default_gated(static_kwargs: dict[str, Any], dynamic_args: frozenset[str]) -> Reversibility:
    """Conservative default: an unclassified mutation forces approval."""
    return Reversibility.GATED


def _default_reversible(
    static_kwargs: dict[str, Any], dynamic_args: frozenset[str]
) -> Reversibility:
    """Creating a soft-deletable entity is undone by deleting it: reversible."""
    return Reversibility.REVERSIBLE


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
    # Optional create surface: set ``create_adapter`` to emit a {resource}.create op.
    create_adapter: Callable[..., Awaitable[Any]] | None = None  # async (ctx, **fields) -> row
    creatable: frozenset[str] = field(default_factory=frozenset)  # type: ignore[reportUnknownVariableType]
    create_classify: ClassifyFn = _default_reversible  # create reversibility (delete undoes it)
    inverse_create: InverseFn | None = None  # defaults to {resource}.delete on the created id
    # Optional restore surface: set ``restore_adapter`` to emit a {resource}.restore op.
    # This is what makes the default delete inverse ({resource}.restore) executable —
    # without it, a soft-delete records an inverse op that resolves to nothing.
    restore_adapter: Callable[..., Awaitable[Any]] | None = None  # async (ctx, id) -> row
    restore_classify: ClassifyFn = _default_reversible  # restore reversibility (delete undoes it)
    inverse_restore: InverseFn | None = None  # defaults to {resource}.delete on the id
    restore_action_kind: str = "restore"
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


def _make_inverse_create(spec: EntitySpec) -> InverseFn:
    def inverse_create(
        kwargs: dict[str, Any], before: dict[str, Any] | None, after: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        # Reverse a create by deleting the created row. Skip when the adapter
        # returned no id (e.g. a dedup hit that reused an existing row, which
        # must not be deleted) — such adapters should supply ``inverse_create``.
        if not after or "id" not in after:
            return None
        return {"op": f"{spec.resource}.delete", "id": str(after["id"])}

    return inverse_create


def _make_inverse_restore(spec: EntitySpec) -> InverseFn:
    def inverse_restore(
        kwargs: dict[str, Any], before: dict[str, Any] | None, after: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        # Reverse a restore by soft-deleting the row again.
        return {"op": f"{spec.resource}.delete", "id": str(kwargs["id"])}

    return inverse_restore


def build_crud_specs(spec: EntitySpec) -> list[OpSpec]:
    """Emit the four uniform CRUD OpSpecs for one declared entity.

    get/search are reads; patch/delete carry snapshot + inverse + classify so the
    approval scanner grades them and the ledger can reverse them.
    """
    search_view = spec.search_view or spec.view
    snapshot = spec.snapshot or _make_snapshot(spec)
    inverse_patch = spec.inverse_patch or _make_inverse_patch(spec)
    inverse_delete = spec.inverse_delete or _make_inverse_delete(spec)
    inverse_create = spec.inverse_create or _make_inverse_create(spec)

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

    specs = [
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

    # Create is opt-in: only entities that declare a ``create_adapter`` get a
    # {resource}.create op. The adapter owns the create (flush so the view sees
    # the id); the generic wrapper projects the view and records the inverse.
    if spec.create_adapter is not None:
        create_adapter = spec.create_adapter

        async def _create(ctx: Any, **fields: Any) -> dict[str, Any]:
            return _to_dict(spec.view(await create_adapter(ctx, **fields)))

        specs.append(
            OpSpec(
                name=f"{spec.resource}.create",
                apply=_create,
                subject_type=spec.subject_type,
                patchable=spec.creatable,
                inverse=inverse_create,
                classify=spec.create_classify,
            )
        )

    # Restore is opt-in: only entities that declare a ``restore_adapter`` get a
    # {resource}.restore op. The adapter loads the soft-deleted row and clears
    # its tombstone; the generic wrapper projects the restored view and records
    # the inverse (delete again). This is the executable counterpart to the
    # default delete inverse, which already names {resource}.restore.
    if spec.restore_adapter is not None:
        restore_adapter = spec.restore_adapter
        inverse_restore = spec.inverse_restore or _make_inverse_restore(spec)

        async def _restore(ctx: Any, *, id: Any) -> dict[str, Any]:
            return _to_dict(spec.view(await restore_adapter(ctx, id)))

        specs.append(
            OpSpec(
                name=f"{spec.resource}.restore",
                apply=_restore,
                action_kind=spec.restore_action_kind,
                subject_type=spec.subject_type,
                inverse=inverse_restore,
                classify=spec.restore_classify,
            )
        )

    return specs
