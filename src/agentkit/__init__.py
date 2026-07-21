"""agentkit — domain-blind agent runtime."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:
    # Single source of truth: the installed package metadata (pyproject `version`).
    # Avoids the string here drifting out of sync with every release bump.
    __version__ = _pkg_version("agentkit")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0+unknown"

from agentkit.compaction import COMPACTION_SUMMARY_ANNOTATION, CompactionResult, compact_history
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
from agentkit.pii import Action as PiiAction
from agentkit.pii import (
    Detector,
    Firewall,
    PiiPolicy,
    RehydratePolicy,
    Span,
    TokenMap,
    ZdrRouteUnavailable,
    wrap_provider,
)
from agentkit.providers.base import RoutingPreferences
from agentkit.session import AgentSession

__all__ = [
    "COMPACTION_SUMMARY_ANNOTATION",
    "Action",
    "AgentConfig",
    "AgentSession",
    "CompactionResult",
    "Detector",
    "Envelope",
    "Firewall",
    "PendingConfirmation",
    "PiiAction",
    "PiiPolicy",
    "RehydratePolicy",
    "RoutingPreferences",
    "Span",
    "TokenMap",
    "ToolCallSummary",
    "ValidationResult",
    "Violation",
    "ZdrRouteUnavailable",
    "__version__",
    "compact_history",
    "validate_envelope",
    "wrap_provider",
]
