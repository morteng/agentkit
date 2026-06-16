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


def _names_in(target: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(target) if isinstance(n, ast.Name)}


# Node types whose target(s) rebind a name in a way we can't treat as a constant.
# `.target` covers AnnAssign/AugAssign/For/comprehension/NamedExpr.
_TARGET_REBIND = (
    ast.AnnAssign,
    ast.AugAssign,
    ast.For,
    ast.AsyncFor,
    ast.comprehension,
    ast.NamedExpr,
)


def _rebound_names(node: ast.AST) -> set[str]:
    """Names a node binds in a non-constant way (anything but ``Name = literal``)."""
    if isinstance(node, _TARGET_REBIND):
        return _names_in(node.target)
    if isinstance(node, (ast.With, ast.AsyncWith)):
        return {
            name
            for item in node.items
            if item.optional_vars is not None
            for name in _names_in(item.optional_vars)
        }
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        a = node.args
        return {
            arg.arg
            for arg in (*a.posonlyargs, *a.args, *a.kwonlyargs, a.vararg, a.kwarg)
            if arg is not None
        }
    if isinstance(node, (ast.Global, ast.Nonlocal)):
        return set(node.names)
    return set()


def _const_bindings(tree: ast.AST) -> dict[str, Any]:
    """Names that provably hold a single literal value for the whole script.

    Resolves the common idiom ``s = "review"; ...patch(status=s)`` so a variable
    carrying a free-transition literal is not over-gated, while ``s = "published"``
    still gates. A name qualifies only when it is assigned exactly once, by a
    plain ``Name = <literal>`` statement, and is never rebound any other way
    (reassignment, augmented assign, loop/with/comprehension target, walrus,
    function parameter, global/nonlocal). Anything ambiguous is left out, so the
    registry falls back to its conservative dynamic-arg gate — safety is never
    weakened, only false over-gating of provable constants is removed.
    """
    literal_assigns: dict[str, list[Any]] = {}
    disqualified: set[str] = set()

    for node in ast.walk(tree):
        # Only a single bare-Name target with a literal value is a candidate;
        # tuple unpacking / chained targets disqualify every name involved.
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            name = node.targets[0].id
            ok, val = _static_value(node.value)
            (literal_assigns.setdefault(name, []).append(val) if ok else disqualified.add(name))
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                disqualified |= _names_in(t)
        else:
            disqualified |= _rebound_names(node)

    return {
        name: vals[0]
        for name, vals in literal_assigns.items()
        if name not in disqualified and len(vals) == 1
    }


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
        consts = _const_bindings(tree)
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
                if not ok and isinstance(kw.value, ast.Name) and kw.value.id in consts:
                    ok, val = True, consts[kw.value.id]
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
