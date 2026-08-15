"""PII firewall policy + the fail-closed routing exceptions."""

from typing import Literal

from pydantic import BaseModel, Field


class PiiPolicy(BaseModel):
    """Per-workspace firewall policy. Consumer-supplied.

    Defaults are the safe baseline: no employer-name scrubbing (that is
    semantically-required public data and an explicit opt-in), require a
    zero-data-retention route, EU residency off, no blocked models, and an
    explicit "do not collect" preference on every PII-carrying request.
    """

    scrub_employer_names: bool = False
    require_zdr: bool = True
    eu_only: bool = False

    blocked_models: set[str] = Field(default_factory=set)  # type: ignore[reportUnknownVariableType]
    """Model ids this workspace must never send PII to.

    Enforced by ``ScrubbingProvider`` on the *active* path only: a request the
    firewall is inert for carries no PII and is none of this policy's business.
    A blocked model raises :class:`BlockedModelError`.
    """

    default_data_collection: Literal["deny", "allow", "unset"] = "deny"
    """The data-collection preference sent when ``require_zdr`` is off.

    A request the firewall is active on is, by definition, carrying PII. Saying
    nothing about data handling on such a request does not put "no preference"
    on the wire — it takes the provider's default, and provider defaults permit
    retention and training. ``"deny"`` therefore asks explicitly for no
    collection while leaving fallbacks enabled, so routing degrades instead of
    failing.

    ``"allow"`` states the opposite preference explicitly (maximum provider
    availability). ``"unset"`` restores the older behaviour of sending no
    routing block at all when neither ``require_zdr`` nor ``eu_only`` is set.
    """


class ZdrRouteUnavailable(Exception):
    """Raised when a PII-flagged call cannot be routed to a compliant
    (zero-data-retention / no-train) provider.

    The firewall fails closed rather than silently downgrading to a
    non-compliant route.
    """


class BlockedModelError(Exception):
    """Raised when a PII-carrying request targets a model in ``blocked_models``.

    ``blocked_models`` was a policy field nothing ever read: the firewall would
    happily send PII to a model the operator had explicitly forbidden. It is
    now enforced at the egress boundary, before the request is scrubbed or
    sent.
    """
