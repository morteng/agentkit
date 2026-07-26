"""Validate tool call arguments against a :class:`ToolSpec`'s JSON Schema.

Used at the in-process MCP boundary (``InProcessMCPClient.call_tool``) to
catch malformed arguments *before* they reach the handler. A stdio/subprocess
MCP server validates independently inside the real MCP SDK; the in-process
shortcut bypasses that transport entirely, so this is the only place those
handlers get schema enforcement.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from jsonschema import Draft7Validator

from agentkit.tools.spec import ToolError, ToolSpec

if TYPE_CHECKING:
    from jsonschema.exceptions import ValidationError


def validate_arguments(spec: ToolSpec, arguments: dict[str, Any]) -> ToolError | None:
    """Validate ``arguments`` against ``spec.parameters`` (a JSON Schema).

    Returns ``None`` when the arguments are valid, or when the spec declares
    no meaningful schema (empty ``parameters``). On failure, returns a
    :class:`ToolError` naming every offending field and what was expected —
    structured enough for the model to self-correct on the next call — in
    place of an opaque handler-side exception (``TypeError``, ``KeyError``,
    ...) for a malformed dict.
    """
    schema = spec.parameters
    if not schema:
        return None
    validator = Draft7Validator(schema)
    errors: list[ValidationError] = sorted(
        validator.iter_errors(arguments),  # type: ignore[reportUnknownMemberType]
        key=lambda e: list(e.path),
    )
    if not errors:
        return None
    details = "; ".join(_describe(e) for e in errors)
    return ToolError(
        code="invalid_arguments",
        message=f"Invalid arguments for '{spec.name}': {details}",
        retryable=True,
    )


def _describe(error: ValidationError) -> str:
    """Render one ``ValidationError`` as a short, field-anchored explanation."""
    field = ".".join(str(p) for p in error.path) or "(root)"
    if error.validator == "required":
        # error.instance is the object missing one or more of the required
        # keys; error.validator_value is the full required list. Diff them so
        # the message names only what is actually absent, not the whole list.
        # Both are typed `Any | Unset` in the jsonschema stubs (the sentinel
        # covers construction paths this library never takes), but the JSON
        # Schema "required" keyword guarantees list[str] / Mapping[str, Any]
        # for any error the validator itself produces.
        required = cast("list[str]", error.validator_value)
        instance = cast("dict[str, Any]", error.instance)
        missing = sorted(set(required) - set(instance))
        return f"missing required field(s): {', '.join(missing)}"
    if error.validator == "type":
        return (
            f"field '{field}': expected type {error.validator_value!r}, "
            f"got {type(error.instance).__name__} ({error.instance!r})"
        )
    if error.validator == "enum":
        return f"field '{field}': must be one of {error.validator_value!r}, got {error.instance!r}"
    return f"field '{field}': {error.message}"
