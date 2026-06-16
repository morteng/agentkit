"""OpRegistry — the single store of OpSpecs, read by namespace, scanner, ledger."""

from __future__ import annotations

from typing import Any

from agentkit.resources.types import OpSpec, Reversibility


class OpRegistry:
    def __init__(self) -> None:
        self._specs: dict[str, OpSpec] = {}

    def register(self, spec: OpSpec) -> None:
        self._specs[spec.name] = spec

    def get(self, name: str) -> OpSpec:
        return self._specs[name]

    def has(self, name: str) -> bool:
        return name in self._specs

    def classify(
        self, name: str, static_kwargs: dict[str, Any], dynamic_args: frozenset[str]
    ) -> Reversibility:
        """Grade one call. Unknown ops and read-less specs default to GATED
        (conservative); reads are REVERSIBLE; writes defer to spec.classify."""
        spec = self._specs.get(name)
        if spec is None:
            return Reversibility.GATED
        if spec.is_read:
            return Reversibility.REVERSIBLE
        if spec.classify is None:
            return Reversibility.GATED
        return spec.classify(static_kwargs, dynamic_args)
