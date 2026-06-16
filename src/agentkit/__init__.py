"""agentkit — domain-blind agent runtime."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:
    # Single source of truth: the installed package metadata (pyproject `version`).
    # Avoids the string here drifting out of sync with every release bump.
    __version__ = _pkg_version("agentkit")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0+unknown"

from agentkit.config import AgentConfig
from agentkit.envelope import (
    Action,
    Envelope,
    PendingConfirmation,
    ToolCallSummary,
    ValidationResult,
    Violation,
)
from agentkit.finalize_validator import validate_envelope
from agentkit.session import AgentSession

__all__ = [
    "Action",
    "AgentConfig",
    "AgentSession",
    "Envelope",
    "PendingConfirmation",
    "ToolCallSummary",
    "ValidationResult",
    "Violation",
    "__version__",
    "validate_envelope",
]
