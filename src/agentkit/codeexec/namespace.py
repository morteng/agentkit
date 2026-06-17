"""Restricted builtins and stdout capture for the executor."""

from __future__ import annotations

import builtins as _builtins
import collections as _collections
import datetime as _datetime
import decimal as _decimal
import itertools as _itertools
import json as _json
import math as _math
import re as _re
import statistics as _statistics
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from types import ModuleType

# Pure-compute stdlib modules a host MAY merge into the script namespace so
# scripts can do real math/date/parsing work WITHOUT an `import` statement
# (imports stay banned by the validator). These are stateless and expose no
# filesystem, network, process, or introspection surface. Dunder attribute
# access on them is still rejected at parse time by the validator, so handing
# the real module objects to a script does not reopen the sandbox escape.
#
# Deliberately excluded: os, sys, subprocess, importlib, pathlib, socket,
# builtins, and anything else with IO / process / import reach. `random` is
# left out too (carries mutable global state).
SAFE_MODULES: dict[str, ModuleType] = {
    "math": _math,
    "statistics": _statistics,
    "datetime": _datetime,
    "json": _json,
    "decimal": _decimal,
    "itertools": _itertools,
    "collections": _collections,
    "re": _re,
}

SAFE_BUILTIN_NAMES = (
    "len",
    "range",
    "enumerate",
    "zip",
    "sorted",
    "reversed",
    "sum",
    "min",
    "max",
    "abs",
    "round",
    "list",
    "dict",
    "set",
    "tuple",
    "str",
    "int",
    "float",
    "bool",
    "isinstance",
    "any",
    "all",
    "map",
    "filter",
    "repr",
    "frozenset",
    "bytes",
    # Iterator protocol — `next(iter(xs))` is the idiomatic "first match" and a
    # common need in generated data scripts; both are pure and non-escaping.
    "iter",
    "next",
    # Type inspection. The AST validator rejects dunder names and getattr/eval
    # are not exposed, so `type(x)` cannot be walked to __subclasses__/__globals__
    # — the only forms that would make it a sandbox-escape primitive.
    "type",
    # Pure numeric / encoding helpers — no I/O, no attribute traversal.
    "divmod",
    "pow",
    "chr",
    "ord",
    "hex",
    "oct",
    "bin",
    "format",
    "hash",
    "True",
    "False",
    "None",
    # Builtin exception classes — needed so generated code can write defensive
    # `try/except ValueError` and `raise ValueError(...)`. Without these the
    # interpreter raises `NameError: name 'ValueError' is not defined` the moment
    # a script references them, which makes recoverable error handling
    # impossible. Same safety reasoning as `type`: the AST validator rejects
    # dunder access, so an exception class cannot be walked to
    # __subclasses__/__globals__ — exposing the names adds no escape surface.
    "Exception",
    "BaseException",
    "ValueError",
    "TypeError",
    "KeyError",
    "IndexError",
    "AttributeError",
    "RuntimeError",
    "LookupError",
    "ArithmeticError",
    "ZeroDivisionError",
    "OverflowError",
    "StopIteration",
    "StopAsyncIteration",
    "AssertionError",
    "NotImplementedError",
)


class StdoutBuffer:
    """Bounded text buffer for captured print() output."""

    def __init__(self, max_bytes: int) -> None:
        self._max = max_bytes
        self._parts: list[str] = []
        self._size = 0
        self._truncated = False

    def write(self, text: str) -> None:
        if self._truncated:
            return
        remaining = self._max - self._size
        if len(text) > remaining:
            self._parts.append(text[:remaining])
            self._parts.append("\n…[truncated]")
            self._truncated = True
        else:
            self._parts.append(text)
            self._size += len(text)

    def getvalue(self) -> str:
        return "".join(self._parts)


def build_safe_builtins(buffer: StdoutBuffer) -> dict[str, Any]:
    """A __builtins__ dict containing only whitelisted names + a captured print."""
    safe: dict[str, Any] = {}
    for name in SAFE_BUILTIN_NAMES:
        if hasattr(_builtins, name):
            safe[name] = getattr(_builtins, name)

    def _print(*args: object, sep: str = " ", end: str = "\n", **_ignored: object) -> None:
        buffer.write(sep.join(str(a) for a in args) + end)

    safe["print"] = _print
    return safe
