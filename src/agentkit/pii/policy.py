"""PII firewall policy + the fail-closed routing exception."""

from pydantic import BaseModel, Field


class PiiPolicy(BaseModel):
    """Per-workspace firewall policy. Consumer-supplied.

    Defaults are the safe baseline: no employer-name scrubbing (that is
    semantically-required public data and an explicit opt-in), require a
    zero-data-retention route, EU residency off, no blocked models.
    """

    scrub_employer_names: bool = False
    require_zdr: bool = True
    eu_only: bool = False
    blocked_models: set[str] = Field(default_factory=set)  # type: ignore[reportUnknownVariableType]


class ZdrRouteUnavailable(Exception):
    """Raised when a PII-flagged call cannot be routed to a compliant
    (zero-data-retention / no-train) provider.

    The firewall fails closed rather than silently downgrading to a
    non-compliant route.
    """
