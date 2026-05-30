"""Resource limits for a single script run."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExecLimits:
    wall_clock_s: float = 30.0
    max_stdout_bytes: int = 64 * 1024
