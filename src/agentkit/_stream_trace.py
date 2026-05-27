"""Per-session raw stream-delta logger for chat truncation investigations.

This is the agentkit-side companion to Pikkolo's
``api/app/agents/sdk/stream_trace.py``. It exists because chat truncation
bugs (Pikkolo F2: leading-character drops at chunk boundaries) need a
checkpoint UPSTREAM of the Pikkolo agentkit adapter — the existing
``adapter_in`` trace already shows the chars missing, so the bug enters
before agentkit hands the delta over.

The two modules share a contract (env vars + JSONL shape) so traces from
both sides land in the same directory and can be diffed by ``session_id``
+ ``ts``. Each side reads the env independently; neither imports the
other.

Opt-in via env:

    STREAM_TRACE_SESSIONS=<sid>,<sid>,*    # allowlist or wildcard
    STREAM_TRACE_DIR=/tmp/some/path        # output dir (default below)

Without ``STREAM_TRACE_SESSIONS`` the tracer is a no-op (one set lookup).
Failures during write (disk full, permission denied) are swallowed —
instrumentation MUST never break the streaming hot path.

JSONL line shape (one object per delta):

    {
      "ts": "2026-05-27T10:14:33.421Z",
      "session_id": "...",
      "checkpoint": "translator_in",   # Pikkolo also uses "adapter_in" / "adapter_out_cleaned"
      "content_repr": "'Her er hva...'",  # repr() so whitespace stays visible
      "content_len": 22,
      "iteration": 3,                     # optional
      "extra": {...},                     # optional, caller-supplied
    }
"""

from __future__ import annotations

import contextlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any

# Default trace dir matches Pikkolo's existing tooling so traces from both
# sides land in the same place without configuration. Ops can override
# with STREAM_TRACE_DIR if a different layout is needed (Ampæra, future
# tenants).
_DEFAULT_TRACE_DIR = "/tmp/pikkolo-stream-trace"

_SESSIONS_ENV = "STREAM_TRACE_SESSIONS"
_DIR_ENV = "STREAM_TRACE_DIR"

_raw_env = os.environ.get(_SESSIONS_ENV, "")
_trace_all_sessions: bool = any(s.strip() == "*" for s in _raw_env.split(","))
_traced_sessions: frozenset[str] = frozenset(
    s.strip().lower() for s in _raw_env.split(",") if s.strip() and s.strip() != "*"
)

_TRACE_DIR = Path(os.environ.get(_DIR_ENV, _DEFAULT_TRACE_DIR))

# Process-local lock keeps JSONL lines atomic when two coroutines on the
# same event loop emit at the same microsecond. Cross-process atomicity is
# provided by POSIX append-mode opens (writes up to PIPE_BUF are atomic).
_write_lock = Lock()


def is_tracing(session_id: str | None) -> bool:
    """Cheap membership check. Callers gate format work behind this."""
    if not session_id:
        return False
    if _trace_all_sessions:
        return True
    if not _traced_sessions:
        return False
    return str(session_id).lower() in _traced_sessions


def trace_delta(
    session_id: str | None,
    checkpoint: str,
    content: str,
    *,
    iteration: int | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Append a JSONL line for one stream delta.

    No-op when the session isn't allowlisted. Disk errors are swallowed so
    instrumentation can never break the chat path.
    """
    if not is_tracing(session_id):
        return

    record: dict[str, Any] = {
        "ts": datetime.now(UTC).isoformat(),
        "session_id": str(session_id),
        "checkpoint": checkpoint,
        "content_repr": repr(content),
        "content_len": len(content),
    }
    if iteration is not None:
        record["iteration"] = iteration
    if extra:
        record["extra"] = extra

    # Disk full, permission denied, mounted-readonly — drop the record;
    # instrumentation MUST NOT break the chat path.
    with contextlib.suppress(OSError):
        _write_record(record)


def _write_record(record: dict[str, Any]) -> None:
    """Append one JSON line. Split out so tests can monkeypatch a failure."""
    _TRACE_DIR.mkdir(parents=True, exist_ok=True)
    sid = record["session_id"]
    path = _TRACE_DIR / f"{sid}.jsonl"
    line = json.dumps(record, ensure_ascii=False)
    with _write_lock, path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
