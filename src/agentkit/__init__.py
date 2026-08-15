"""agentkit — domain-blind agent runtime."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:
    # Single source of truth: the installed package metadata (pyproject `version`).
    # Avoids the string here drifting out of sync with every release bump.
    __version__ = _pkg_version("agentkit")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0+unknown"

from agentkit._content import Provenance
from agentkit.audit import (
    ACTION_APPROVAL_RESOLVED,
    ACTION_TOOL_CALL,
    SOURCE_AGENT,
    SOURCE_HUMAN,
    AuditRecord,
    AuditSink,
    LoggingAuditSink,
    MultiAuditSink,
    NullAuditSink,
)
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
from agentkit.history import (
    DEFAULT_HISTORY_PAGE_LIMIT,
    MAX_HISTORY_PAGE_LIMIT,
    AssistantHistoryItem,
    HistoryPage,
    ToolHistoryItem,
    UserHistoryItem,
    build_history_items,
    load_history_page,
)
from agentkit.pii import Action as PiiAction
from agentkit.pii import (
    BlockedModelError,
    CompositeDetector,
    Detector,
    FieldContextDetector,
    Firewall,
    PiiPolicy,
    RehydratePolicy,
    SecretDetector,
    SecretPolicy,
    Span,
    TokenMap,
    ZdrRouteUnavailable,
    default_detector,
    wrap_provider,
)
from agentkit.providers.base import RoutingPreferences
from agentkit.session import (
    DEFAULT_HISTORY_LIMIT,
    SYNTHETIC_TOOL_RESULT_ANNOTATION,
    AgentSession,
)

__all__ = [
    "ACTION_APPROVAL_RESOLVED",
    "ACTION_TOOL_CALL",
    "COMPACTION_SUMMARY_ANNOTATION",
    "DEFAULT_HISTORY_LIMIT",
    "DEFAULT_HISTORY_PAGE_LIMIT",
    "MAX_HISTORY_PAGE_LIMIT",
    "SOURCE_AGENT",
    "SOURCE_HUMAN",
    "SYNTHETIC_TOOL_RESULT_ANNOTATION",
    "Action",
    "AgentConfig",
    "AgentSession",
    "AssistantHistoryItem",
    "AuditRecord",
    "AuditSink",
    "BlockedModelError",
    "CompactionResult",
    "CompositeDetector",
    "Detector",
    "Envelope",
    "FieldContextDetector",
    "Firewall",
    "HistoryPage",
    "LoggingAuditSink",
    "MultiAuditSink",
    "NullAuditSink",
    "PendingConfirmation",
    "PiiAction",
    "PiiPolicy",
    "Provenance",
    "RehydratePolicy",
    "RoutingPreferences",
    "SecretDetector",
    "SecretPolicy",
    "Span",
    "TokenMap",
    "ToolCallSummary",
    "ToolHistoryItem",
    "UserHistoryItem",
    "ValidationResult",
    "Violation",
    "ZdrRouteUnavailable",
    "__version__",
    "build_history_items",
    "compact_history",
    "default_detector",
    "load_history_page",
    "validate_envelope",
    "wrap_provider",
]
