"""Scriptable resource framework — uniform CRUD verbs, reversibility registry,
approval scanner. Domain-free; the consuming app supplies OpSpecs."""

from agentkit.resources.registry import OpRegistry
from agentkit.resources.scanner import ApprovalScanner
from agentkit.resources.types import (
    OpSpec,
    Reversibility,
    ScanFinding,
    ScriptClassification,
)

__all__ = [
    "ApprovalScanner",
    "OpRegistry",
    "OpSpec",
    "Reversibility",
    "ScanFinding",
    "ScriptClassification",
]
