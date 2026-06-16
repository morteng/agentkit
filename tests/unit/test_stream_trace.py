"""Tests for the per-session stream-delta tracer.

Tracer is meant for chat truncation investigations (Pikkolo F2 etc.) and
must never affect the streaming hot path. These tests pin:

  - opt-in via STREAM_TRACE_SESSIONS env: empty -> no-op, allowlist -> match,
    "*" -> match every session.
  - JSONL on-disk shape: stable keys, repr-encoded content (whitespace and
    unicode control chars stay visible), session-id-per-file.
  - OSError on write is swallowed (instrumentation must never break callers).

The tracer reads the env at import time, so each test imports the module
fresh via importlib.reload after mutating os.environ.
"""

import importlib
import json


def _reload_module(monkeypatch, **env):
    """Reload agentkit._stream_trace with the given env. Returns the module."""
    for key in ("STREAM_TRACE_SESSIONS", "STREAM_TRACE_DIR"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    import agentkit._stream_trace as mod

    return importlib.reload(mod)


def test_is_tracing_false_when_env_unset(monkeypatch):
    mod = _reload_module(monkeypatch)
    assert mod.is_tracing("any-session-id") is False
    assert mod.is_tracing(None) is False
    assert mod.is_tracing("") is False


def test_is_tracing_allowlist_match(monkeypatch):
    sid = "abc-123"
    mod = _reload_module(monkeypatch, STREAM_TRACE_SESSIONS=f" {sid} , other-sid ")
    assert mod.is_tracing(sid) is True
    assert mod.is_tracing(sid.upper()) is True  # case-insensitive match
    assert mod.is_tracing("not-listed") is False


def test_is_tracing_wildcard_matches_every_session(monkeypatch):
    mod = _reload_module(monkeypatch, STREAM_TRACE_SESSIONS="*")
    assert mod.is_tracing("anything") is True
    assert mod.is_tracing("e2e-fcd-93") is True


def test_trace_delta_writes_jsonl_when_session_allowlisted(monkeypatch, tmp_path):
    sid = "session-under-trace"
    mod = _reload_module(
        monkeypatch,
        STREAM_TRACE_SESSIONS=sid,
        STREAM_TRACE_DIR=str(tmp_path),
    )

    mod.trace_delta(sid, "translator_in", "Hei ")
    mod.trace_delta(sid, "translator_in", "verden\n")

    path = tmp_path / f"{sid}.jsonl"
    assert path.exists(), "tracer must create per-session file under STREAM_TRACE_DIR"
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    rec0 = json.loads(lines[0])
    assert rec0["session_id"] == sid
    assert rec0["checkpoint"] == "translator_in"
    # repr() so trailing whitespace and control chars stay diff-visible
    assert rec0["content_repr"] == repr("Hei ")
    assert rec0["content_len"] == 4
    assert "ts" in rec0


def test_trace_delta_noop_when_not_allowlisted(monkeypatch, tmp_path):
    mod = _reload_module(
        monkeypatch,
        STREAM_TRACE_SESSIONS="only-this-one",
        STREAM_TRACE_DIR=str(tmp_path),
    )
    mod.trace_delta("a-different-session", "translator_in", "should not appear")
    assert list(tmp_path.iterdir()) == [], "no file should be created for unlisted sessions"


def test_trace_delta_swallows_oserror(monkeypatch, tmp_path):
    """A failing write must not propagate — instrumentation MUST NOT break callers."""
    mod = _reload_module(
        monkeypatch,
        STREAM_TRACE_SESSIONS="*",
        STREAM_TRACE_DIR=str(tmp_path),
    )

    def _boom(_record):
        raise PermissionError("disk full / read-only mount")

    # Patch the writer entry point on the freshly-reloaded module.
    monkeypatch.setattr(mod, "_write_record", _boom)
    # Should not raise.
    mod.trace_delta("sid", "translator_in", "x")


def test_extra_field_preserved(monkeypatch, tmp_path):
    sid = "sid-with-extras"
    mod = _reload_module(
        monkeypatch,
        STREAM_TRACE_SESSIONS="*",
        STREAM_TRACE_DIR=str(tmp_path),
    )
    mod.trace_delta(sid, "translator_in", "x", iteration=3, extra={"model": "test/m"})

    rec = json.loads((tmp_path / f"{sid}.jsonl").read_text().strip())
    assert rec["iteration"] == 3
    assert rec["extra"] == {"model": "test/m"}
