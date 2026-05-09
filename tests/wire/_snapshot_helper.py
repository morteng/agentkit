"""Tiny snapshot helper backed by JSON files.

Tests call ``assert_event_snapshot(event, snapshot_name)`` which writes to
``tests/wire/snapshots/<snapshot_name>.json`` on first run (or when
``WIRE_SNAPSHOT_UPDATE=1`` is set) and asserts equality otherwise.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

SNAPSHOT_DIR = Path(__file__).parent / "snapshots"


def _normalize(value: Any) -> Any:
    """Drop fields that vary run-to-run (timestamps, ULIDs) so snapshots are stable."""
    if isinstance(value, dict):
        return {
            k: _normalize(v)
            for k, v in value.items()
            if k not in {"created_at", "timestamp", "id", "trace_id"}
        }
    if isinstance(value, list):
        return [_normalize(v) for v in value]
    return value


def assert_event_snapshot(event_payload: dict[str, Any], snapshot_name: str) -> None:
    path = SNAPSHOT_DIR / f"{snapshot_name}.json"
    normalized = _normalize(event_payload)
    rendered = json.dumps(normalized, indent=2, sort_keys=True, ensure_ascii=False)

    if os.environ.get("WIRE_SNAPSHOT_UPDATE") == "1" or not path.exists():
        SNAPSHOT_DIR.mkdir(exist_ok=True)
        path.write_text(rendered + "\n")
        return

    actual = rendered + "\n"
    expected = path.read_text()
    if actual != expected:
        raise AssertionError(
            f"Wire snapshot drift for {snapshot_name!r}.\n"
            f"Expected (from {path}):\n{expected}\n"
            f"Got:\n{actual}\n"
            f"If this change is intentional, re-run with WIRE_SNAPSHOT_UPDATE=1."
        )
