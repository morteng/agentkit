"""Restricted builtins and stdout capture for the executor."""

from __future__ import annotations

import builtins as _builtins
from typing import Any

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
    "True",
    "False",
    "None",
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
