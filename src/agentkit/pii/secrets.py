"""High-entropy secret detection — the credential half of the PII firewall.

The firewall's other recognizers are consumer-supplied and identity-shaped
(national ID numbers, bank accounts, names, emails). Credentials are *not*
identity-shaped: a generated password is a random string with no format at
all, so a pattern list can never see it. This module closes that hole.

The motivating incident: a user-provisioning tool minted a household password
with ``secrets.token_urlsafe(16)`` and returned it inside its tool result. The
result went straight back to the model, out to a third-party inference
provider on the same turn, and into the transcript forever. Twenty-two
characters of base64url match no identity pattern, so nothing stopped it.

What is detected
----------------
* **PEM private-key blocks** — ``-----BEGIN … PRIVATE KEY-----`` through
  ``-----END … PRIVATE KEY-----``, at any length.
* **Vendor-prefixed keys** — a deliberately small, documented set of prefixes
  that are unambiguous in the wild (see :data:`VENDOR_PATTERNS`). This is not
  meant to be exhaustive; the entropy path is the general net.
* **JWTs** — three base64url segments separated by dots, anchored on the
  ``eyJ`` header that a JSON header always base64-encodes to.
* **High-entropy runs** — base64/base64url tokens and long hex strings, scored
  with a real Shannon-entropy calculation plus length and character-class
  heuristics rather than a shape regex, so novel formats are caught too.
* **Secret-named fields** — when the firewall knows the JSON field a string
  came from (``arguments={"password": …}``), a field named for a credential
  redacts its whole value regardless of entropy. This is the rule that catches
  low-entropy human-chosen passwords, which entropy alone cannot.

The false-positive tradeoff
---------------------------
Redacting the wrong thing is not free: a redacted git SHA breaks a build
report, a redacted UUID breaks a lookup the model then retries. The thresholds
below were tuned against a corpus of things that *look* random but are not
(``tests/unit/pii/test_secret_detector.py``). The deliberate choices:

* **Digest-shaped hex is exempt by default.** A 40-char hex string is a git
  SHA far more often than it is an API key, and the two are indistinguishable
  in isolation. Hex runs of exactly 32/40/56/64/96/128 characters (md5, sha1,
  sha224, sha256, sha384, sha512, and their git-object cousins) are passed
  through unless a credential keyword sits next to them. Set
  :attr:`SecretPolicy.exempt_digest_shaped_hex` to ``False`` to trade that
  precision for recall. Hex of any other length ≥
  :attr:`SecretPolicy.hex_min_length` is treated as a token.
* **UUIDs and ULIDs are exempt** by shape. Both are high-entropy by
  construction and are identifiers, not credentials — and agentkit mints ULIDs
  for every message and session, so they are everywhere in tool output.
* **Runs longer than :attr:`SecretPolicy.max_length` (512) are exempt.**
  Credentials are short; base64-encoded images, certificates and documents are
  long. Payloads directly following a ``base64,`` data-URI marker are skipped
  at any length. PEM keys are matched by their delimiters, so the length cap
  never hides one.
* **At least two of {lowercase, uppercase, digit} are required** on the
  entropy path. This drops ordinary long words, snake_case identifiers and
  all-lowercase slugs, at the cost of missing single-case random strings —
  which a random-token generator essentially never produces at these lengths.
* **Word-shaped runs are rejected structurally**, not by entropy. Entropy
  cannot separate ``TheQuickBrownFoxJumpsOverTheLazyDog`` (4.63 bits/char, more
  than a random token) from a credential. Two shape tests do: a cap on the
  longest single-character-class run, and the mean length of the word chunks
  the run tiles into (:func:`mean_word_chunk`). Concatenated words and
  acronyms — ``AbstractSingletonProxyFactoryBean``, ``PostgreSQLJDBCDriver``,
  ``Screenshot_2026-08-15_at_14`` — tile into long chunks; random tokens shred
  into one- and two-character ones.
* **Keyword context relaxes everything.** Within
  :attr:`SecretPolicy.context_window` characters after ``password``,
  ``api_key``, ``authorization`` and friends, the bar drops to a short,
  modestly-random run — because in that position a random-looking string
  almost certainly is one.

Every threshold above is a field on :class:`SecretPolicy`.

Measured recall, and where it runs out
--------------------------------------
Measured over thousands of generated tokens, *unlabelled* — no field name, no
keyword anywhere near them:

===========================  ======  ==============
value                        chars   missed
===========================  ======  ==============
``token_urlsafe(16)``        22      ~1.2 %
``token_urlsafe(24)``        32      ~0.9 %
``token_urlsafe(32)``        43      ~1.4 %
``token_urlsafe(12)``        16      ~21 %
``token_urlsafe(8)``         11      100 % (under ``min_length``)
``token_hex(16)`` / ``(32)`` 32 / 64 100 % (digest-shaped, see above)
===========================  ======  ==============

Label the value and the table stops applying. Under a credential-named field
the miss rate is zero at every length, ``token_urlsafe(8)`` included, because
the field rule redacts the value outright rather than scoring it. Next to a
credential keyword in free text it is zero from 16 characters up. That is the
shape credentials almost always arrive in — it is the shape the incident above
had — so the unlabelled path is the backstop, not the primary net.

Also out of scope by construction: weak human-chosen passwords in unlabelled
free text (no entropy to find), classic base64 alphabets where ``+`` or ``/``
splits a value into sub-16-character pieces, and single-case random strings.

Usage
-----
``SecretDetector`` is a plain :class:`~agentkit.pii.protocols.Detector` and is
default-on inside :func:`~agentkit.pii.composite.default_detector`. Compose it
with domain recognizers instead of replacing it::

    from agentkit.pii import CompositeDetector, Firewall, PiiPolicy

    detector = CompositeDetector.with_defaults(MyNorwegianDetector())
    firewall = Firewall(detector, PiiPolicy())
"""

from __future__ import annotations

import math
import re
from collections import Counter

from pydantic import BaseModel, Field

from agentkit.pii.spans import merge_spans
from agentkit.pii.types import Action, Span

#: Substrings that mark the text *before* a candidate as credential context.
#: Kept narrow on purpose: bare ``key`` and bare ``auth`` match ``keyboard``
#: and ``author``, so they are not here.
SECRET_CONTEXT_KEYWORDS: tuple[str, ...] = (
    "password",
    "passwd",
    "passphrase",
    "pwd",
    "secret",
    "token",
    "api_key",
    "apikey",
    "api-key",
    "access_key",
    "accesskey",
    "private_key",
    "privatekey",
    "client_secret",
    "credential",
    "authorization",
    "bearer",
    "signing_key",
    "session_key",
    "otp",
    "one_time_code",
    "recovery_code",
)

#: Field-name components that make a whole value a secret regardless of how it
#: scores. ``{"password": "sommer2026"}`` is a credential even though the value
#: has no entropy to speak of.
SECRET_FIELD_NAMES: frozenset[str] = frozenset(
    {
        "password",
        "passwd",
        "pwd",
        "passphrase",
        "secret",
        "token",
        "apikey",
        "credential",
        "credentials",
        "privatekey",
        "otp",
        "authorization",
    }
)

#: Name components that turn a credential word into a measurement:
#: ``token_count``, ``password_length``, ``secret_budget``. An LLM runtime is
#: full of these and their values are numbers, not credentials.
NON_SECRET_FIELD_COMPONENTS: frozenset[str] = frozenset(
    {
        "count",
        "counts",
        "limit",
        "length",
        "len",
        "size",
        "usage",
        "budget",
        "num",
        "total",
        "max",
        "min",
        "estimate",
        "cost",
        "price",
        "policy",
        "type",
        "kind",
        "name",
        "expiry",
        "expires",
        "ttl",
    }
)

#: Values that carry no secret even under a credential-named field.
_EMPTY_VALUE_RE = re.compile(r"\A(true|false|none|null|nil|n/?a|-+|0)\Z", re.IGNORECASE)

#: Field names ending in one of these are secrets too — ``admin_password``,
#: ``stripe_api_key``, ``refresh_token``.
SECRET_FIELD_SUFFIXES: tuple[str, ...] = (
    "password",
    "passwd",
    "passphrase",
    "secret",
    "token",
    "apikey",
    "api_key",
    "access_key",
    "accesskey",
    "private_key",
    "privatekey",
    "credential",
    "credentials",
)

#: Vendor key formats with prefixes unambiguous enough to match on sight. A
#: small, curated set — the entropy path is what catches everything else.
#:
#: ``sk-``            OpenAI / Anthropic (``sk-``, ``sk-proj-``, ``sk-ant-``)
#: ``ghp_`` &c        GitHub personal/OAuth/user/server/refresh tokens
#: ``github_pat_``    GitHub fine-grained PAT
#: ``xox[abprs]-``    Slack bot/user/app/refresh tokens
#: ``AKIA``/``ASIA``  AWS access key IDs (long-term / STS)
#: ``AIza``           Google API keys
#: ``[sr]k_live_``    Stripe secret / restricted keys
VENDOR_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "SECRET_PEM",
        re.compile(
            r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY( BLOCK)?-----"
            r"[\s\S]*?"
            r"-----END [A-Z0-9 ]*PRIVATE KEY( BLOCK)?-----"
        ),
    ),
    (
        "SECRET_JWT",
        re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{4,}"),
    ),
    ("SECRET_API_KEY", re.compile(r"\bsk-[A-Za-z0-9_-]{16,}")),
    ("SECRET_GITHUB_TOKEN", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{16,}")),
    ("SECRET_GITHUB_TOKEN", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}")),
    ("SECRET_SLACK_TOKEN", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}")),
    ("SECRET_AWS_KEY_ID", re.compile(r"\b(?:AKIA|ASIA|AGPA|AIDA|AROA|ANPA|ANVA)[A-Z0-9]{16}\b")),
    ("SECRET_GOOGLE_API_KEY", re.compile(r"\bAIza[A-Za-z0-9_-]{35}\b")),
    ("SECRET_STRIPE_KEY", re.compile(r"\b[sr]k_(?:live|test)_[A-Za-z0-9]{16,}\b")),
)

#: Runs of base64url-safe characters, with optional base64 padding. ``+`` and
#: ``/`` are deliberately excluded: including them merges file paths
#: (``/usr/local/lib/python3``) into single high-scoring runs.
_CANDIDATE_RE = re.compile(r"[A-Za-z0-9_-]{6,}={0,2}")

_UUID_RE = re.compile(
    r"\A[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\Z"
)
#: Crockford base32, 26 chars, first char 0-7 — the ULID agentkit mints for
#: every session, turn and message id.
_ULID_RE = re.compile(r"\A[0-7][0-9ABCDEFGHJKMNPQRSTVWXYZ]{25}\Z")
#: A placeholder a previous scrub already wrote. Keeps scrubbing idempotent.
_PLACEHOLDER_RE = re.compile(r"\A\[[A-Z_]+(_\d+)?\]\Z")

#: Word-shaped chunks: an all-caps acronym, a capitalised or lowercase word, or
#: a digit group. ``PostgreSQLJDBCDriverManager`` tiles into four long chunks;
#: a random token shreds into one- and two-character ones.
_CHUNK_RE = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|[0-9]+")

_HEX_CHARS = frozenset("0123456789abcdefABCDEF")
#: Output lengths of the common digest functions, in hex characters.
_DIGEST_LENGTHS = frozenset({32, 40, 56, 64, 96, 128})

_DATA_URI_MARKER = "base64,"
_FIELD_SPLIT_RE = re.compile(r"[^a-z0-9]+")

#: A run must mix at least this many of {lowercase, uppercase, digit}. One
#: class is a word, a slug or an all-caps constant.
_MIN_CHAR_CLASSES = 2


def shannon_entropy(text: str) -> float:
    """Shannon entropy of ``text`` in bits per character.

    ``0.0`` for the empty string and for any single repeated character; the
    maximum for a string of length *n* is ``log2(n)``, reached when every
    character is distinct.
    """
    if not text:
        return 0.0
    n = len(text)
    return -sum((c / n) * math.log2(c / n) for c in Counter(text).values())


def _char_class(ch: str) -> str:
    if ch.islower():
        return "lower"
    if ch.isupper():
        return "upper"
    if ch.isdigit():
        return "digit"
    return "symbol"


def alpha_num_classes(text: str) -> set[str]:
    """The subset of ``{lower, upper, digit}`` present in ``text``.

    Symbols are excluded on purpose: ``application_json_content_type`` would
    otherwise clear a "two character classes" bar on its underscores alone.
    """
    return {c for c in (_char_class(ch) for ch in text) if c != "symbol"}


def longest_class_run(text: str) -> int:
    """Length of the longest run of consecutive same-class characters.

    The structural discriminator between a random token (which switches class
    every few characters) and a name like ``getUserProfileByIdentifier`` or
    ``Screenshot_2026-08-15`` (which has long homogeneous stretches).
    """
    best = 0
    run = 0
    previous = ""
    for ch in text:
        current = _char_class(ch)
        run = run + 1 if current == previous else 1
        previous = current
        best = max(best, run)
    return best


def mean_word_chunk(text: str) -> float:
    """Mean length of the word-shaped chunks ``text`` tiles into.

    The sharpest single discriminator between concatenated words and random
    bytes. ``PostgreSQLJDBCDriverManager`` → ``Postgre|SQLJDBC|Driver|Manager``,
    mean 6.75. ``U-sOgGhuMRf5xSS0XI7gPQ`` → fifteen chunks, mean 1.5. English
    identifiers sit above 3.5 almost without exception; fewer than one random
    token in two hundred does.

    ``0.0`` when the text contains no letters or digits at all.
    """
    chunks = _CHUNK_RE.findall(text)
    if not chunks:
        return 0.0
    return sum(len(c) for c in chunks) / len(chunks)


class SecretPolicy(BaseModel):
    """Thresholds for :class:`SecretDetector`. Every knob is a tradeoff dial.

    The defaults favour precision on unlabelled text and recall on text that
    names its credential. See the module docstring for the reasoning.
    """

    action: Action = Action.NEVER_SEND
    """What the firewall does with a hit.

    ``NEVER_SEND`` (default) drops the value: never sent, never tokenized,
    never written to the durable token map. ``TOKENIZE`` replaces it with a
    stable placeholder that the consumer's own display path can rehydrate —
    useful when the human on the other end genuinely needs to read the
    generated password, at the cost of storing a live credential in the token
    map.
    """

    min_length: int = 16
    """Shortest run considered on the unlabelled entropy path."""

    max_length: int = 512
    """Longest run considered. Above this a run is treated as encoded content
    (image, certificate, document), not a credential. PEM blocks are matched by
    their delimiters and ignore this cap."""

    entropy_bits: float = 3.6
    """Entropy floor, in bits per character, for a run mixing letters and digits."""

    uniform_entropy_bits: float = 4.0
    """Higher floor for a run with no digits — mixed-case letters only, where
    natural-language identifiers live."""

    max_class_run: int = 8
    """Reject a run containing a longer stretch of one character class."""

    word_chunk_length: float = 3.5
    """Reject a run whose mean word-chunk length reaches this — the signature
    of concatenated words and acronyms (``AbstractSingletonProxyFactoryBean``).
    Applied only to runs containing lowercase letters, since an all-caps random
    token legitimately tiles into one long chunk. See :func:`mean_word_chunk`."""

    hex_min_length: int = 32
    """Shortest all-hex run considered."""

    hex_entropy_bits: float = 3.0
    """Entropy floor for hex runs. Hex tops out at 4.0 bits/char, so this is
    the equivalent of ``entropy_bits`` on a 16-symbol alphabet."""

    exempt_digest_shaped_hex: bool = True
    """Pass through hex runs whose length matches a common digest (32/40/56/
    64/96/128) unless credential context sits next to them. Turn off to catch
    32- and 64-hex API keys at the cost of redacting checksums and git SHAs."""

    context_window: int = 48
    """How many characters before a run are searched for a credential keyword."""

    context_min_length: int = 8
    """Shortest run considered when credential context is present."""

    context_entropy_bits: float = 3.0
    """Entropy floor when credential context is present."""

    context_keywords: tuple[str, ...] = SECRET_CONTEXT_KEYWORDS
    """Keywords that put a run in credential context."""

    redact_secret_named_fields: bool = True
    """Redact the entire value of a field named for a credential, whatever it
    contains. Only applies when the caller supplies a field name (the firewall
    does so while walking tool arguments and JSON tool results)."""

    detect_vendor_prefixes: bool = True
    """Match the curated :data:`VENDOR_PATTERNS` set."""

    detect_entropy: bool = True
    """Run the general entropy path. Off leaves only PEM/vendor/field rules."""

    allow_patterns: tuple[str, ...] = Field(default=())
    """Regexes for values that must never be flagged. A candidate run is
    exempt when it matches one in full (``re.fullmatch``). Use for
    domain-specific identifiers that look random — order numbers, device ids."""

    model_config = {"frozen": True}


class SecretDetector:
    """Detect credential-shaped strings in text.

    Implements :class:`~agentkit.pii.protocols.Detector` and the optional
    :class:`~agentkit.pii.protocols.FieldContextDetector` extension, so the
    firewall can hand it the JSON field name a string came from.

    Stateless and thread-safe once constructed; the compiled allow-patterns are
    the only per-instance state.
    """

    def __init__(self, policy: SecretPolicy | None = None) -> None:
        self.policy = policy or SecretPolicy()
        self._allow = [re.compile(p) for p in self.policy.allow_patterns]

    # ---- Detector ----------------------------------------------------------

    def detect(self, text: str) -> list[Span]:
        return self.detect_in_field(text, None)

    def detect_in_field(self, text: str, field: str | None) -> list[Span]:
        """Detect with the surrounding field name as extra context.

        ``field`` is the JSON key the string was found under, if any. A
        credential-named field both relaxes the thresholds for the whole string
        and (under :attr:`SecretPolicy.redact_secret_named_fields`) redacts the
        value outright.
        """
        if not text:
            return []
        policy = self.policy
        field_is_secret = field is not None and is_secret_field_name(field)

        if field_is_secret and policy.redact_secret_named_fields:
            stripped = text.strip()
            if not stripped or _PLACEHOLDER_RE.match(stripped) or _EMPTY_VALUE_RE.match(stripped):
                return []
            return [Span(start=0, end=len(text), kind="SECRET", action=policy.action)]

        spans: list[Span] = []
        if policy.detect_vendor_prefixes:
            for kind, pattern in VENDOR_PATTERNS:
                for match in pattern.finditer(text):
                    spans.append(
                        Span(start=match.start(), end=match.end(), kind=kind, action=policy.action)
                    )
        if policy.detect_entropy:
            spans.extend(self._entropy_spans(text, field_is_secret=field_is_secret))
        return merge_spans(spans)

    # ---- Entropy path ------------------------------------------------------

    def _entropy_spans(self, text: str, *, field_is_secret: bool) -> list[Span]:
        spans: list[Span] = []
        lowered = text.lower()
        for match in _CANDIDATE_RE.finditer(text):
            start, run = _trim_separators(match.start(), match.group(0))
            if not run:
                continue
            if _follows_data_uri(text, start):
                continue
            in_context = field_is_secret or self._has_context(lowered, start)
            kind = self._classify(run, in_context=in_context)
            if kind is not None:
                spans.append(
                    Span(start=start, end=start + len(run), kind=kind, action=self.policy.action)
                )
        return spans

    def _has_context(self, lowered_text: str, start: int) -> bool:
        window = lowered_text[max(0, start - self.policy.context_window) : start]
        return any(keyword in window for keyword in self.policy.context_keywords)

    def _classify(self, run: str, *, in_context: bool) -> str | None:
        """Return the span kind for ``run``, or ``None`` when it is not a secret."""
        if not self._is_candidate(run):
            return None
        entropy = shannon_entropy(run)
        if _is_hex(run):
            return self._classify_hex(run, entropy, in_context=in_context)
        return self._classify_base64(run, entropy, in_context=in_context)

    def _is_candidate(self, run: str) -> bool:
        """Cheap exclusions: encoded blobs, known identifier shapes, allow-list."""
        if len(run) > self.policy.max_length:
            return False
        if _UUID_RE.match(run) or _ULID_RE.match(run) or _PLACEHOLDER_RE.match(run):
            return False
        return not any(p.fullmatch(run) for p in self._allow)

    def _classify_base64(self, run: str, entropy: float, *, in_context: bool) -> str | None:
        policy = self.policy
        n = len(run)
        classes = alpha_num_classes(run)
        if len(classes) < _MIN_CHAR_CLASSES:
            return None
        if in_context:
            labelled = n >= policy.context_min_length and entropy >= policy.context_entropy_bits
            return "SECRET" if labelled else None
        if n < policy.min_length or longest_class_run(run) > policy.max_class_run:
            return None
        if any(ch.islower() for ch in run) and mean_word_chunk(run) >= policy.word_chunk_length:
            return None
        floor = policy.entropy_bits if "digit" in classes else policy.uniform_entropy_bits
        return "SECRET" if entropy >= floor else None

    def _classify_hex(self, run: str, entropy: float, *, in_context: bool) -> str | None:
        policy = self.policy
        n = len(run)
        if in_context:
            if n >= policy.context_min_length and entropy >= policy.context_entropy_bits:
                return "SECRET_HEX"
            return None
        if n < policy.hex_min_length or entropy < policy.hex_entropy_bits:
            return None
        if policy.exempt_digest_shaped_hex and n in _DIGEST_LENGTHS:
            return None
        return "SECRET_HEX"


def is_secret_field_name(field: str) -> bool:
    """True when a JSON field name says its value is a credential.

    Matches a whole ``snake_case``/``camelCase``/``kebab-case`` component
    against :data:`SECRET_FIELD_NAMES`, or the flattened name against
    :data:`SECRET_FIELD_SUFFIXES` — so ``password``, ``adminPassword`` and
    ``stripe-api-key`` all match while ``keyboard_layout`` does not.
    """
    normalized = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", field).lower()
    parts = {p for p in _FIELD_SPLIT_RE.split(normalized) if p}
    if parts & NON_SECRET_FIELD_COMPONENTS:
        return False
    if parts & SECRET_FIELD_NAMES:
        return True
    flattened = normalized.replace("-", "_")
    return flattened.endswith(SECRET_FIELD_SUFFIXES)


def _is_hex(run: str) -> bool:
    return all(ch in _HEX_CHARS for ch in run)


def _trim_separators(start: int, run: str) -> tuple[int, str]:
    """Drop leading/trailing separators so ``_2026-`` scores as ``2026``.

    Trailing ``=`` is kept — it is base64 padding, part of the value.
    """
    lead = len(run) - len(run.lstrip("-_"))
    trimmed = run[lead:].rstrip("-_")
    return start + lead, trimmed


def _follows_data_uri(text: str, start: int) -> bool:
    """True when the run is the payload of a ``data:…;base64,`` URI."""
    prefix = text[max(0, start - len(_DATA_URI_MARKER)) : start]
    return prefix == _DATA_URI_MARKER
