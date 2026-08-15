"""Canonical tool types — provider-agnostic."""

from collections.abc import Iterable
from enum import StrEnum
from typing import Any, Literal, cast

from pydantic import BaseModel, Field

from agentkit._content import Provenance
from agentkit.pii.types import RehydratePolicy

DEFAULT_TOOL_TIMEOUT_SECONDS = 60.0
"""Fallback wall-clock budget for a single tool invocation.

Applied by :meth:`agentkit.tools.registry.ToolRegistry.invoke` when a spec
declares no usable ``timeout_seconds`` (``None`` or non-positive). A hung MCP
server must never hang the turn.
"""


class RiskLevel(StrEnum):
    READ = "read"
    LOW_WRITE = "low_write"
    HIGH_WRITE = "high_write"
    DESTRUCTIVE = "destructive"


class SideEffects(StrEnum):
    NONE = "none"
    LOCAL = "local"
    EXTERNAL_REVERSIBLE = "external_reversible"
    EXTERNAL_IRREVERSIBLE = "external_irreversible"


class ApprovalPolicy(StrEnum):
    NEVER = "never"  # always auto-approve
    BY_RISK = "by_risk"  # let ApprovalGate decide based on risk
    ALWAYS = "always"  # always require user approval


class ContentBlockOut(BaseModel):
    """Content block returned by a tool (text or image)."""

    type: Literal["text", "image"]
    text: str | None = None
    image_url: str | None = None
    image_data: str | None = None
    media_type: str | None = None


class ToolError(BaseModel):
    """Structured error info on tool failure."""

    code: str
    message: str
    retryable: bool = False


class ToolSpec(BaseModel):
    """Provider-agnostic tool definition.

    The ``name`` is the auto-namespaced identifier (e.g. ``acme.devices.list``,
    ``kit.finalize``). Registry assigns the prefix; tool authors pass the bare name.
    """

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema
    returns: dict[str, Any] | None = None  # JSON Schema for result, optional
    risk: RiskLevel
    idempotent: bool
    side_effects: SideEffects
    requires_approval: ApprovalPolicy
    cache_ttl_seconds: int | None
    timeout_seconds: float
    rehydrate: RehydratePolicy = RehydratePolicy.DENY
    """PII rehydration policy for this tool's model-produced arguments.

    Default ``DENY`` keeps placeholders as placeholders (backward-compatible and
    safe — closes the exfiltration channel). Only artifact-render / form-fill
    tools with no model-supplied destination fields should set ``ALLOW``.
    """


class ToolCall(BaseModel):
    id: str  # provider's tool_use_id
    name: str
    arguments: dict[str, Any]


class ToolResult(BaseModel):
    call_id: str
    status: Literal["ok", "error", "denied", "timeout", "cancelled"]
    content: list[ContentBlockOut] = Field(default_factory=list)  # type: ignore[reportUnknownVariableType]
    error: ToolError | None = None
    duration_ms: int = 0
    cached: bool = False
    provenance: Provenance = Provenance.SYSTEM
    """Trust level of the content this result carries.

    Defaults to ``SYSTEM``, so a tool written before provenance existed keeps
    its previous (trusted) treatment. A tool that hands the model third-party
    text — web fetch, inbox read, another tenant's records — must set
    ``Provenance.UNTRUSTED``: that is what taints the turn and disables write
    actions for the rest of it. See :mod:`agentkit.guards.taint`.
    """


def unknown_tool_message(name: str, known_names: Iterable[str] | None = None) -> str:
    """The error text for a call to a name no registered tool matches.

    A dotted name (``content.get``, ``tasks.patch``) is almost always a
    scripting-namespace method the model reached for as a standalone tool. Naming
    that explicitly lets the model self-correct in one hop — call it inside the
    scripting tool, or use the matching flat tool if one exists — instead of
    ping-ponging on a bare "unknown tool".

    A plain (un-dotted) name that suffix-matches exactly ONE registered
    qualified name — ``web_search`` when only ``acme.web_search`` is
    registered — is almost always the model copying an advertised tool minus
    its namespace prefix. Surface a "did you mean" pointing at the routable
    qualified name so the model retries with a name the registry will accept.
    This is a suggestion only: no transparent rerouting. When ``known_names``
    is not supplied, or the suffix match is absent/ambiguous, plain names get
    the bare message.
    """
    if "." in name:
        return (
            f"unknown tool: {name}. Dotted names are scripting-namespace methods, "
            f"not standalone tools — call them inside the scripting tool "
            f"(await {name}(...)), or use the matching flat tool if one exists."
        )
    if known_names is not None:
        # Namespacing convention is ``<server>.<tool>`` (split on the first
        # dot); mirror plane.py's ``_bare`` so the suffix compared here is the
        # same one advertised elsewhere.
        matches = [k for k in known_names if k.split(".", 1)[-1] == name]
        if len(matches) == 1:
            return f"unknown tool: {name}. Did you mean {matches[0]}?"
    return f"unknown tool: {name}"


# Mapping from JSON Schema primitive type name to the Python types that
# satisfy it. ``integer``/``number`` are handled specially so ``True`` — an
# ``int`` subclass in Python — does not pass as a number.
_JSON_TYPE_CHECKS: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
    "null": (type(None),),
}


def _type_matches(json_type: str, value: Any) -> bool:
    expected = _JSON_TYPE_CHECKS.get(json_type)
    if expected is None:
        return True  # unknown/extension type keyword — do not second-guess it
    if json_type in ("integer", "number") and isinstance(value, bool):
        return False
    return isinstance(value, expected)


def validate_tool_arguments(schema: dict[str, Any] | None, arguments: dict[str, Any]) -> list[str]:
    """Structurally validate model-produced ``arguments`` against a tool schema.

    Deliberately *not* a JSON Schema implementation: ``jsonschema`` is not a
    dependency of this library and adding a runtime dependency for a
    pre-dispatch sanity check is not worth it. What is checked is the subset
    that catches the mistakes models actually make and that no reasonable
    schema disagrees about:

    * every ``required`` property is present;
    * no unexpected top-level keys when ``additionalProperties`` is ``false``;
    * declared top-level ``type`` keywords match the value's Python type
      (including ``type`` given as a list of alternatives).

    Nested objects, ``enum``, ``format``, and the combinators are left to the
    tool itself — a conservative check must never reject a call a real
    validator would accept. Returns the list of problems, empty when the
    arguments are acceptable.
    """
    if schema is None:
        return []
    declared_type = schema.get("type")
    if declared_type is not None and declared_type != "object":
        return []  # not an object schema — nothing structural to say

    problems: list[str] = []

    raw_required = schema.get("required")
    if isinstance(raw_required, list):
        required = cast("list[Any]", raw_required)
        missing = [str(k) for k in required if str(k) not in arguments]
        if missing:
            problems.append(f"missing required argument(s): {', '.join(sorted(missing))}")

    raw_properties = schema.get("properties")
    properties: dict[str, Any] = (
        cast("dict[str, Any]", raw_properties) if isinstance(raw_properties, dict) else {}
    )

    if schema.get("additionalProperties") is False and properties:
        unexpected = [k for k in arguments if k not in properties]
        if unexpected:
            problems.append(
                f"unexpected argument(s): {', '.join(sorted(unexpected))}. "
                f"Allowed: {', '.join(sorted(properties))}"
            )

    for key, value in arguments.items():
        prop = properties.get(key)
        if not isinstance(prop, dict):
            continue
        expected = cast("dict[str, Any]", prop).get("type")
        if isinstance(expected, str):
            if not _type_matches(expected, value):
                problems.append(
                    f"argument {key!r} must be of type {expected}, got {type(value).__name__}"
                )
        elif isinstance(expected, list) and expected:
            alternatives = [str(t) for t in cast("list[Any]", expected)]
            if not any(_type_matches(t, value) for t in alternatives):
                problems.append(
                    f"argument {key!r} must be one of type {', '.join(alternatives)}, "
                    f"got {type(value).__name__}"
                )

    return problems


def invalid_arguments_message(name: str, problems: Iterable[str]) -> str:
    """Model-readable text for a call whose arguments failed validation."""
    joined = "; ".join(problems)
    return (
        f"invalid arguments for {name}: {joined}. "
        f"The call was not executed — retry with arguments matching the tool's schema."
    )
