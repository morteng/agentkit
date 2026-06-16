"""Scriptable resource framework — uniform CRUD verbs, reversibility registry,
approval scanner. Domain-free; the consuming app supplies OpSpecs."""

from agentkit.resources.entity import EntitySpec, build_crud_specs
from agentkit.resources.namespace import ResourceNamespace
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
    "EntitySpec",
    "OpRegistry",
    "OpSpec",
    "ResourceNamespace",
    "Reversibility",
    "ScanFinding",
    "ScriptClassification",
    "build_crud_specs",
]
