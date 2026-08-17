"""Exception hierarchy. All agentkit exceptions inherit from AgentkitError."""

from collections.abc import Mapping, Sequence


class AgentkitError(Exception):
    """Base for every exception raised by this library."""


class ConfigurationError(AgentkitError):
    """Misconfiguration detected at startup or session creation."""


class InvalidPhaseTransition(AgentkitError):
    """The loop attempted a transition not in the transition table."""

    def __init__(self, from_: str, to: str) -> None:
        super().__init__(f"Invalid phase transition: {from_} -> {to}")
        self.from_ = from_
        self.to = to


class ProviderError(AgentkitError):
    """Underlying LLM provider returned an error."""


class ToolError(AgentkitError):
    """Tool dispatch or execution failed."""


class ApprovalTimeout(AgentkitError):
    """Approval was requested but no decision came in time.

    ``call_ids`` are the approvals the expiry stranded, so the caller can emit
    one resolution event per card instead of a bare error the client cannot
    match to anything it is showing.

    ``tool_names`` maps those ids to what each call would have run. Expiry is
    the one path that emits no ``ApprovalGranted``/``ApprovalDenied``, and it
    clears ``pending_user_approvals`` as it goes — so unless the names travel
    on this exception they are gone by the time the resolution events are
    built, and an expiry notice can name only an opaque id. Ids absent from
    the mapping resolve to :data:`~agentkit.events.approval.UNNAMED_TOOL`.

    ``sequence_base`` is the turn's next unused ``event_sequence`` at the
    moment the timeout was detected — the checkpoint's ``ctx`` is still in
    scope at the raise site, so the caller building the timeout stream can
    continue the turn's ``(turn_id, sequence)`` numbering instead of
    restarting at 0 and colliding with events the suspended turn already
    emitted (e.g. ``TurnStarted``, sequence 0).
    """

    def __init__(
        self,
        message: str,
        *,
        call_ids: Sequence[str] = (),
        tool_names: Mapping[str, str] | None = None,
        sequence_base: int = 0,
    ) -> None:
        super().__init__(message)
        self.call_ids: tuple[str, ...] = tuple(call_ids)
        self.tool_names: dict[str, str] = dict(tool_names or {})
        self.sequence_base = sequence_base


class CheckpointMissing(AgentkitError):
    """resume_with_approval called but the checkpoint is gone."""


class StoreError(AgentkitError):
    """Storage backend operation failed."""
