"""Resource limits for a single script run."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExecLimits:
    """Resource limits applied to a single ``execute()`` call.

    Note: wall_clock_s cannot interrupt a synchronous CPU loop or an injected
    coroutine that swallows CancelledError; see executor module docstring.
    """

    wall_clock_s: float = 30.0
    max_stdout_bytes: int = 64 * 1024
