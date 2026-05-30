"""Async, in-process curated-namespace executor.

The body is reparented under an `async def` via AST so the script may use
`await` and `return`. Runs under a wall-clock timeout with a restricted
__builtins__ and only the caller-supplied namespace.
"""

from __future__ import annotations

import ast
import asyncio
import time
from dataclasses import dataclass
from typing import Any

from agentkit.codeexec.errors import CodeExecutionError, CodeTimeoutError, CodeValidationError
from agentkit.codeexec.limits import ExecLimits
from agentkit.codeexec.namespace import StdoutBuffer, build_safe_builtins
from agentkit.codeexec.validator import validate_source

_MAIN = "__agentkit_main__"


@dataclass
class ExecutionResult:
    stdout: str
    return_value: Any
    error: str | None
    error_type: str | None
    duration_ms: int


async def execute(
    namespace: dict[str, Any],
    source: str,
    limits: ExecLimits | None = None,
) -> ExecutionResult:
    limits = limits or ExecLimits()
    buffer = StdoutBuffer(max_bytes=limits.max_stdout_bytes)
    started = time.monotonic()

    try:
        tree = validate_source(source)
    except CodeValidationError as exc:
        return _result(buffer, None, exc, started)

    # Build `async def __agentkit_main__(): <body>` via AST to allow await + return.
    # ast.AsyncFunctionDef constructor args aren't fully typed in typeshed; use
    # keyword form and suppress the one narrowing diagnostic.
    func: ast.AsyncFunctionDef = ast.AsyncFunctionDef(  # pyright: ignore[reportCallIssue]
        name=_MAIN,
        args=ast.arguments(
            posonlyargs=[],
            args=[],
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[],
        ),
        body=tree.body,
        decorator_list=[],
        returns=None,
    )
    module = ast.Module(body=[func], type_ignores=[])
    ast.fix_missing_locations(module)

    g: dict[str, Any] = dict(namespace)
    g["__builtins__"] = build_safe_builtins(buffer)

    try:
        code_obj = compile(module, "<script>", "exec")
        exec(code_obj, g)  # defines _MAIN in g
        main: Any = g[_MAIN]
        return_value: Any = await asyncio.wait_for(main(), timeout=limits.wall_clock_s)
    except TimeoutError:
        return _result(buffer, None, CodeTimeoutError(f"exceeded {limits.wall_clock_s}s"), started)
    except CodeExecutionError as exc:
        return _result(buffer, None, exc, started)
    except Exception as exc:
        return _result(buffer, None, exc, started)

    return ExecutionResult(
        stdout=buffer.getvalue(),
        return_value=return_value,
        error=None,
        error_type=None,
        duration_ms=int((time.monotonic() - started) * 1000),
    )


def _result(buffer: StdoutBuffer, rv: Any, exc: Exception, started: float) -> ExecutionResult:
    return ExecutionResult(
        stdout=buffer.getvalue(),
        return_value=rv,
        error=str(exc),
        error_type=type(exc).__name__,
        duration_ms=int((time.monotonic() - started) * 1000),
    )
