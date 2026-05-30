"""AST allowlist validation — the executor's security boundary.

Rejects: import statements, dynamic-eval / introspection builtins by name,
and any dunder attribute access (the standard sandbox-escape vector).
"""

from __future__ import annotations

import ast

from agentkit.codeexec.errors import CodeValidationError

FORBIDDEN_NAMES = frozenset(
    {
        "eval",
        "exec",
        "compile",
        "__import__",
        "open",
        "input",
        "globals",
        "locals",
        "vars",
        "getattr",
        "setattr",
        "delattr",
        "dir",
        "breakpoint",
        "memoryview",
        "help",
        "type",
    }
)


def validate_source(source: str) -> ast.Module:
    """Parse and validate. Returns the parsed module on success.

    Raises CodeValidationError on any disallowed construct.
    """
    try:
        tree = ast.parse(source, "<script>", "exec")
    except SyntaxError as exc:
        raise CodeValidationError(f"syntax error: {exc.msg}") from exc

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise CodeValidationError("import statements are not allowed")
        if isinstance(node, ast.Attribute):
            attr = node.attr
            if attr.startswith("__") and attr.endswith("__"):
                raise CodeValidationError(f"dunder attribute access not allowed: {attr}")
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            raise CodeValidationError(f"name not allowed: {node.id}")
    return tree
