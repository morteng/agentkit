"""In-process curated-namespace code executor for agent-authored scripts."""

from agentkit.codeexec.errors import (
    CodeExecutionError,
    CodeTimeoutError,
    CodeValidationError,
)
from agentkit.codeexec.executor import ExecutionResult, execute
from agentkit.codeexec.limits import ExecLimits
from agentkit.codeexec.namespace import SAFE_MODULES

__all__ = [
    "SAFE_MODULES",
    "CodeExecutionError",
    "CodeTimeoutError",
    "CodeValidationError",
    "ExecLimits",
    "ExecutionResult",
    "execute",
]
