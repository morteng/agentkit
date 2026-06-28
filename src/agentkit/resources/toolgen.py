"""Generate a provider ToolSpec from one OpSpec — the flat surface of a
scriptable op. Keeps the flat chat-tool and the script namespace derived from
one declaration so their names and params cannot drift.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agentkit.tools.spec import ApprovalPolicy, RiskLevel, SideEffects, ToolSpec

if TYPE_CHECKING:
    from agentkit.resources.types import OpSpec, Param


def _property(param: Param) -> dict[str, Any]:
    prop: dict[str, Any] = {"type": param.type, "description": param.description}
    if param.enum is not None:
        prop["enum"] = param.enum
    if param.type == "array" and param.items_type is not None:
        prop["items"] = {"type": param.items_type}
    return prop


def op_to_toolspec(spec: OpSpec, *, timeout_seconds: float = 30.0) -> ToolSpec | None:
    """Emit the flat ToolSpec for ``spec``, or None when it is script-only.

    Risk metadata is derived from ``is_read``: reads are non-consequential,
    writes default to low-write with by-risk approval. A consumer that curates
    risk per tool overrides it after generation (e.g. from its own registry).
    """
    if spec.flat_alias is None:
        return None

    properties: dict[str, dict[str, Any]] = {}
    required: list[str] = []
    for key, param in spec.params.items():
        name = param.alias or key
        properties[name] = _property(param)
        if param.required:
            required.append(name)

    if spec.is_read:
        risk, idempotent = RiskLevel.READ, True
        side_effects, approval = SideEffects.NONE, ApprovalPolicy.NEVER
    else:
        risk, idempotent = RiskLevel.LOW_WRITE, False
        side_effects, approval = SideEffects.LOCAL, ApprovalPolicy.BY_RISK

    return ToolSpec(
        name=spec.flat_alias,
        description=spec.description,
        parameters={"type": "object", "properties": properties, "required": required},
        risk=risk,
        idempotent=idempotent,
        side_effects=side_effects,
        requires_approval=approval,
        cache_ttl_seconds=None,
        timeout_seconds=timeout_seconds,
    )
