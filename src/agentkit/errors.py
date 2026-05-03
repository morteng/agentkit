"""Exception hierarchy. All agentkit exceptions inherit from AgentkitError."""


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
    """Approval was requested but no decision came in time."""


class CheckpointMissing(AgentkitError):
    """resume_with_approval called but the checkpoint is gone."""


class StoreError(AgentkitError):
    """Storage backend operation failed."""
