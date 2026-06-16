"""AST allowlist validation — the executor's security boundary.

Rejects: import statements, dynamic-eval / introspection builtins by name,
and any dunder attribute access (the standard sandbox-escape vector).
"""

from __future__ import annotations

import ast

from agentkit.codeexec.errors import CodeValidationError

# Denylist of builtin names that can break the sandbox: dynamic eval/compile,
# I/O, namespace introspection, and attribute traversal (the string-literal
# bypass of the dunder-name check below). `type` is intentionally NOT here: the
# dunder-attribute rule plus the absence of getattr neutralize the
# `type(x).__bases__[0].__subclasses__()` escape, and `type(x)` is a routine
# inspection the curated namespace also exposes. Keep this set disjoint from
# namespace.SAFE_BUILTIN_NAMES (enforced by a unit test) so the two gates agree.
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
            raise CodeValidationError(
                "import statements are not allowed; any modules the host "
                "provides (e.g. math, datetime, json) are already available "
                "by name — use them directly without import"
            )
        if isinstance(node, ast.Attribute):
            attr = node.attr
            if attr.startswith("__") and attr.endswith("__"):
                raise CodeValidationError(f"dunder attribute access not allowed: {attr}")
        if isinstance(node, ast.Name):
            nid = node.id
            if nid.startswith("__") and nid.endswith("__"):
                raise CodeValidationError(f"dunder name not allowed: {nid}")
            if nid in FORBIDDEN_NAMES:
                raise CodeValidationError(f"name not allowed: {nid}")
    return tree
