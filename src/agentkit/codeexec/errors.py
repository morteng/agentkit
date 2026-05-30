"""Errors raised by the code executor."""

from __future__ import annotations


class CodeExecutionError(Exception):
    """Base for all executor errors."""


class CodeValidationError(CodeExecutionError):
    """Source rejected by the AST allowlist before execution."""


class CodeTimeoutError(CodeExecutionError):
    """Script exceeded its wall-clock budget."""
