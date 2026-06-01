"""ApprovalScanner — AST-walk a script, classify each client mutation call.

Domain-free: it recognises calls of the form ``<client_var>.<resource>.<verb>(...)``
and asks the OpRegistry to grade each. A kwarg whose value is not a literal
constant is reported as a *dynamic* arg; the registry's classify decides
whether a dynamic value on a given field escalates to GATED.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING, Any

from agentkit.resources.types import ScanFinding, ScriptClassification

if TYPE_CHECKING:
    from agentkit.resources.registry import OpRegistry


def _static_value(node: ast.AST) -> tuple[bool, Any]:
    """Best-effort literal extraction. Returns (is_static, value)."""
    try:
        return True, ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError):
        return False, None


class ApprovalScanner:
    def __init__(self, client_var: str = "pikkolo") -> None:
        self._var = client_var

    def _op_name(self, func: ast.Attribute) -> str | None:
        # func is `<var>.<resource>.<verb>` -> Attribute(attr=verb,
        # value=Attribute(attr=resource, value=Name(id=var)))
        mid = func.value
        if not isinstance(mid, ast.Attribute):
            return None
        root = mid.value
        if not isinstance(root, ast.Name) or root.id != self._var:
            return None
        return f"{mid.attr}.{func.attr}"

    def scan(self, source: str, registry: OpRegistry) -> ScriptClassification:
        tree = ast.parse(source, "<script>", "exec")
        findings: list[ScanFinding] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            op_name = self._op_name(node.func)
            if op_name is None or not registry.has(op_name):
                continue
            spec = registry.get(op_name)
            if spec.is_read:
                continue
            static_kwargs: dict[str, Any] = {}
            dynamic_args: set[str] = set()
            for kw in node.keywords:
                if kw.arg is None:  # **kwargs splat — unknowable, treat as dynamic
                    dynamic_args.add("**")
                    continue
                ok, val = _static_value(kw.value)
                if ok:
                    static_kwargs[kw.arg] = val
                else:
                    dynamic_args.add(kw.arg)
            rev = registry.classify(op_name, static_kwargs, frozenset(dynamic_args))
            findings.append(
                ScanFinding(
                    op_name=op_name,
                    reversibility=rev,
                    dynamic_args=frozenset(dynamic_args),
                    lineno=node.lineno,
                )
            )
        return ScriptClassification(findings=findings)
