"""Generic, domain-free types for scriptable resource operations.

The consuming application declares one OpSpec per mutating (or read)
operation; agentkit assigns no meaning to resource/field strings beyond
membership and the consumer-supplied classify callable.
"""

from __future__ import annotations

import enum
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any


class Reversibility(enum.Enum):
    """How consequential a mutation is. Drives the approval scanner.

    REVERSIBLE   — ledger-only; runs without prompt even in interactive mode.
    GATED        — restorable but consequential (publish, delete); forces approval.
    IRREVERSIBLE — no inverse (hard delete, external send); forces approval.
    """

    REVERSIBLE = "reversible"
    GATED = "gated"
    IRREVERSIBLE = "irreversible"

    @property
    def severity(self) -> int:
        return {"reversible": 0, "gated": 1, "irreversible": 2}[self.value]


# apply:    async (ctx, **kwargs) -> Any (a typed view, or a dict)
# snapshot: async (ctx, **kwargs) -> dict[str, Any] | None     (before_state; writes only)
# inverse:  (kwargs, before, after) -> dict[str, Any] | None    (ledger inverse-op; writes only)
# classify: (static_kwargs, dynamic_arg_names) -> Reversibility  (writes only)
ApplyFn = Callable[..., Awaitable[Any]]
SnapshotFn = Callable[..., Awaitable[dict[str, Any] | None]]
InverseFn = Callable[
    [dict[str, Any], dict[str, Any] | None, dict[str, Any] | None],
    dict[str, Any] | None,
]
ClassifyFn = Callable[[dict[str, Any], frozenset[str]], Reversibility]


@dataclass
class OpSpec:
    """One scriptable operation, declared once, read by namespace + scanner + ledger."""

    name: str  # "<resource>.<verb>", e.g. "content.patch"
    apply: ApplyFn
    is_read: bool = False
    action_kind: str = "write"  # ledger action_kind for writes
    subject_type: str | None = None
    patchable: frozenset[str] = field(default_factory=frozenset)  # type: ignore[reportUnknownVariableType]
    snapshot: SnapshotFn | None = None
    inverse: InverseFn | None = None
    classify: ClassifyFn | None = None


@dataclass
class ScanFinding:
    """One mutating call found in a script."""

    op_name: str
    reversibility: Reversibility
    dynamic_args: frozenset[str]
    lineno: int


@dataclass
class ScriptClassification:
    """The scanner's verdict for a whole script."""

    findings: list[ScanFinding]

    @property
    def worst(self) -> Reversibility:
        if not self.findings:
            return Reversibility.REVERSIBLE
        return max((f.reversibility for f in self.findings), key=lambda r: r.severity)

    @property
    def requires_approval(self) -> bool:
        return self.worst.severity >= Reversibility.GATED.severity
