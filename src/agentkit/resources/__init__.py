"""Scriptable resource framework — uniform CRUD verbs, reversibility registry,
approval scanner. Domain-free; the consuming app supplies OpSpecs."""

from agentkit.resources.registry import OpRegistry
from agentkit.resources.types import (
    OpSpec,
    Reversibility,
    ScanFinding,
    ScriptClassification,
)

__all__ = [
    "OpRegistry",
    "OpSpec",
    "Reversibility",
    "ScanFinding",
    "ScriptClassification",
]
