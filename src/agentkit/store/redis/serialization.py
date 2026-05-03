"""Versioned JSON serialisation for Redis-backed payloads.

Every payload includes ``_v`` (schema version). Backends running migrations on
read can detect older payloads and upgrade them in place.
"""

import json
from typing import Any


def to_versioned_json(payload: dict[str, Any], *, schema_version: int) -> bytes:
    enriched = {"_v": schema_version, **payload}
    return json.dumps(enriched, separators=(",", ":"), default=str).encode("utf-8")


def from_versioned_json(raw: bytes | str) -> tuple[dict[str, Any], int]:
    """Return ``(payload_without_v, schema_version)``.

    Payloads missing ``_v`` are treated as version 0.
    """
    data = json.loads(raw)
    version = int(data.pop("_v", 0))
    return data, version
