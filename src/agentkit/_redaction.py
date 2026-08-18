"""Projection helpers for values that leave the runtime and reach a human.

Audit records (:mod:`agentkit.audit`) and replayed history
(:mod:`agentkit.history`) both take tool arguments the model wrote — which may
in turn have come from a torrent name, a web page, an inbox — and put them
somewhere a person reads them: a ledger row, a rehydrated page. Both need the
same three properties, and they need to agree on them: secrets masked, nesting
flattened, length bounded. Two copies of that rule drift, and the copy that
drifts is the one that leaks.
"""

import re
import unicodedata
from collections.abc import Mapping
from typing import Any

SECRET_KEY_PATTERN = re.compile(r"(?i)(pass|pwd|token|secret|key|auth|cookie)")
"""Argument/detail keys whose *value* is never recorded. Deliberately broad: a
false positive costs one masked field in a receipt, a false negative writes a
credential into a durable log."""

REDACTED = "***"

#: Cap on how many argument keys a preview carries. Keys are sorted before the
#: cut so the same call always previews the same keys.
MAX_PREVIEW_KEYS = 8

#: Cap on a single previewed string value.
MAX_PREVIEW_VALUE_CHARS = 120

_ELLIPSIS = "…"

_STRIPPED_CATEGORIES = frozenset({"Cc", "Cf"})
"""Unicode categories deleted from every string. ``Cf`` matters as much as
``Cc``: it covers the bidi overrides and zero-width joiners that make a
rendered string lie about its content."""


_URL_QUERY_RE = re.compile(r"(https?://[^\s'\"<>]*)\?[^\s'\"<>]*")
"""A URL and everything after its ``?``. Credentials ride in query strings."""

_BEARER_RE = re.compile(r"(?i)\b(bearer|basic)\s+[A-Za-z0-9._\-+/=]{8,}")

_INLINE_SECRET_RE = re.compile(
    r"(?i)\b([a-z0-9_.-]*(?:pass|pwd|token|secret|key|auth|cookie)[a-z0-9_.-]*)"
    r"\s*[=:]\s*[\"']?([^\s,;)\"'&]{4,})"
)


def scrub_free_text(text: str) -> str:
    """Strip credentials out of a string nobody composed for display.

    ``is_secret_key`` above masks by KEY NAME, which only works when the value
    arrives in a structured field someone already labelled. An exception
    message is not that: it is free text, and the credential is inside it.
    ``str(httpx.HTTPStatusError)`` embeds the full request URL, so a failed
    call carries its own query string — and a ``TOKEN=``-style key-name mask
    never sees it, because there is no key.

    Masking therefore happens on the VALUE. Three shapes, all observed in real
    exception text:

    * ``https://host/path?api_key=…``  -> query string dropped wholesale.
      Nothing downstream needs it, and enumerating which params are secret is
      the mistake (``api_key`` vs ``apikey`` vs ``t``/``s``).
    * ``Authorization: Bearer <token>`` -> the token.
    * ``password=hunter2`` loose in a sentence -> the value.

    This does NOT hide hostnames or paths: they are useful for diagnosis and
    are not credentials. Callers that must not egress an internal hostname need
    a different function, and should say so rather than assuming this one.
    """
    text = _URL_QUERY_RE.sub(r"\1?" + REDACTED, text)
    text = _BEARER_RE.sub(lambda m: f"{m.group(1)} {REDACTED}", text)
    text = _INLINE_SECRET_RE.sub(lambda m: f"{m.group(1)}={REDACTED}", text)
    return text


def is_secret_key(key: str) -> bool:
    return SECRET_KEY_PATTERN.search(key) is not None


def clean_text(text: str, *, limit: int) -> str:
    """Strip control/format characters, collapse whitespace, cap the length.

    The result is display text: it is safe to put in a log line, a JSON field
    or a DOM node as ``textContent``, and it is not the original value. Callers
    that need the original must not go through here.
    """
    stripped = "".join(ch for ch in text if unicodedata.category(ch) not in _STRIPPED_CATEGORIES)
    collapsed = " ".join(stripped.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: max(0, limit - 1)] + _ELLIPSIS


def preview_value(key: str, value: Any) -> Any:
    """One argument value, projected to a flat display-safe scalar."""
    if is_secret_key(key):
        return REDACTED
    if isinstance(value, Mapping):
        return "[object]"
    if isinstance(value, list | tuple | set):
        return "[array]"
    if isinstance(value, str):
        return clean_text(value, limit=MAX_PREVIEW_VALUE_CHARS)
    # bool/int/float/None survive as themselves: they are already scalar, JSON
    # representable, and too small to hide anything in.
    return value


def argument_preview(arguments: Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
    """Project tool arguments to ``(preview, truncated)``.

    Flat and lossy on purpose. A receipt exists so a human can tell *which*
    delete was proposed, not so the full payload survives in a second place —
    and a nested structure rendered from a log is exactly where an injected
    payload gets a second chance at a reader.
    """
    keys = sorted(arguments)
    kept = keys[:MAX_PREVIEW_KEYS]
    preview = {key: preview_value(key, arguments[key]) for key in kept}
    return preview, len(keys) > len(kept)
