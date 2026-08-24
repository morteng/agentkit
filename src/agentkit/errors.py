"""Exception hierarchy. All agentkit exceptions inherit from AgentkitError."""

from collections.abc import Mapping, Sequence

from agentkit._codes import ErrorCode


class AgentkitError(Exception):
    """Base for every exception raised by this library.

    ``code`` is the :class:`~agentkit.events.ErrorCode` the loop puts on the
    :class:`~agentkit.events.Errored` event when this exception ends a turn.

    It exists because ``code`` is the only structured field on that event, and
    the only one a consumer can safely render: ``message`` is free text composed
    on the far side of the wire and may carry a provider's HTTP body, an
    internal hostname or a stack frame. Before this, every raised exception
    reached the consumer as ``INTERNAL`` — a deliberate policy refusal was
    indistinguishable from an unhandled crash, and the only way to tell them
    apart was to pattern-match the exception's name out of ``message``. That
    couples a UI to ``repr`` output, re-breaks whenever wording changes, and
    parses precisely the untrusted field the code/message split exists to keep
    out of the render path.

    Subclasses set it as a class attribute; an instance may override it when the
    same class covers outcomes that differ (a ``ProviderError`` raised for a 429,
    say). ``INTERNAL`` remains the default, so an exception that says nothing
    behaves exactly as it did before.
    """

    code: ErrorCode = ErrorCode.INTERNAL


def error_code_for(exc: BaseException) -> ErrorCode:
    """The ``ErrorCode`` an exception asks for, or ``INTERNAL``.

    Deliberately duck-typed rather than gated on :class:`AgentkitError`: a
    consumer's own exception — one raised from a tool, or from a provider
    decorator it wrote — cannot inherit from a class it does not control, and it
    has as much right to name its failure as agentkit's do. Setting
    ``code = ErrorCode.TOOL_FAULT`` on it is enough.

    The ``isinstance`` check is what makes that safe. ``.code`` is a common
    attribute name on third-party exceptions and it is usually a bare string —
    ``openai.APIStatusError.code`` is ``"not_found"``, for one. Only a real
    ``ErrorCode`` member is honoured, so a foreign ``.code`` is ignored rather
    than coerced into a wrong answer or a ``ValueError`` on a failure path.
    """
    code = getattr(exc, "code", None)
    return code if isinstance(code, ErrorCode) else ErrorCode.INTERNAL


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

    code = ErrorCode.PROVIDER_FAULT


class ToolError(AgentkitError):
    """Tool dispatch or execution failed."""

    code = ErrorCode.TOOL_FAULT


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

    code = ErrorCode.APPROVAL_TIMEOUT

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
