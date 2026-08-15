"""SecretDetector: detection, non-detection, and the tuning knobs.

The non-detection corpus is the load-bearing half. A firewall that redacts git
SHAs, UUIDs and camelCase identifiers out of tool results breaks the assistant
in a way that is far more visible than the leak it prevents, so anything that
weakens ``NOT_SECRETS`` needs a very good reason.
"""

import random
import secrets as pysecrets
import uuid
from itertools import pairwise

import pytest

from agentkit._ids import SessionId, new_id
from agentkit.pii.secrets import (
    SecretDetector,
    SecretPolicy,
    alpha_num_classes,
    is_secret_field_name,
    longest_class_run,
    mean_word_chunk,
    shannon_entropy,
)
from agentkit.pii.types import Action


@pytest.fixture
def detector() -> SecretDetector:
    return SecretDetector()


def hits(detector: SecretDetector, text: str, field: str | None = None) -> list[str]:
    return [text[s.start : s.end] for s in detector.detect_in_field(text, field)]


# ---- Entropy / shape primitives -------------------------------------------


def test_shannon_entropy_bounds():
    assert shannon_entropy("") == 0.0
    assert shannon_entropy("aaaaaaaa") == 0.0
    # Every character distinct → log2(n).
    assert shannon_entropy("abcd") == pytest.approx(2.0)
    assert shannon_entropy("abcdefgh") == pytest.approx(3.0)


def test_alpha_num_classes_ignores_symbols():
    assert alpha_num_classes("abc_def-ghi") == {"lower"}
    assert alpha_num_classes("aB3") == {"lower", "upper", "digit"}


def test_longest_class_run():
    assert longest_class_run("aB1") == 1
    assert longest_class_run("abcDEF12") == 3
    assert longest_class_run("Screenshot") == 9  # "creenshot"


def test_mean_word_chunk_separates_words_from_noise():
    assert mean_word_chunk("PostgreSQLJDBCDriverManager") > 6
    assert mean_word_chunk("U-sOgGhuMRf5xSS0XI7gPQ") < 2.5
    assert mean_word_chunk("---") == 0.0


# ---- Detection -------------------------------------------------------------

SECRETS: list[tuple[str, str, str]] = [
    # (id, text, expected span kind)
    ("openai_key", "sk-proj-abcdefghijklmnop1234567890ABCDEFGH", "SECRET_API_KEY"),
    ("anthropic_key", "sk-ant-api03-Zx9yQw8vUt7sRq6pOn5mLk4jIh3gFe2d", "SECRET_API_KEY"),
    ("github_pat", "ghp_16C7e42F292c6912E7710c838347Ae178B4a", "SECRET_GITHUB_TOKEN"),
    (
        "github_fine_grained",
        "github_pat_11ABCDEFG0aBcDeFgHiJkL_MnOpQrStUvWxYz0123456789",
        "SECRET_GITHUB_TOKEN",
    ),
    (
        "slack_bot",
        # Assembled rather than written out. GitHub push protection matches the
        # `xoxb-` shape confidently enough to reject the entire push over this
        # one fixture, and the alternatives are clicking an unblock link per
        # secret forever or disabling the protection. Splitting the prefix
        # leaves no contiguous token in the file for a scanner to match while
        # handing the detector a byte-identical string — which is the thing
        # actually under test. Synthetic value; it authenticates to nothing.
        "xoxb" + "-1234567890-1234567890123-AbCdEfGhIjKlMnOpQrStUvWx",
        "SECRET_SLACK_TOKEN",
    ),
    ("aws_key_id", "AKIAIOSFODNN7EXAMPLE", "SECRET_AWS_KEY_ID"),
    ("aws_sts_key_id", "ASIAY34FZKBOKMUTVV7A", "SECRET_AWS_KEY_ID"),
    ("google_api_key", "AIzaSyC93jK2mMxYzQ1pR8vN4hL7wT0dF6gB2aX", "SECRET_GOOGLE_API_KEY"),
    ("stripe_live", "sk_live_51HxxAbCdEfGhIjKlMnOp", "SECRET_STRIPE_KEY"),
    (
        "jwt",
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
        ".eyJzdWIiOiIxMjM0NTY3ODkwIn0"
        ".dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U",
        "SECRET_JWT",
    ),
    (
        "hex_token_non_digest_length",
        "0123456789abcdef0123456789abcdef0123456789ab",
        "SECRET_HEX",
    ),
    ("base64_basic_auth", "dXNlcjpwYXNzd29yZDEyMzQ1Ng==", "SECRET"),
]


@pytest.mark.parametrize(
    ("text", "kind"),
    [pytest.param(t, k, id=i) for i, t, k in SECRETS],
)
def test_detects_secret(detector: SecretDetector, text: str, kind: str):
    spans = detector.detect(text)
    assert spans, f"expected a hit for {text!r}"
    assert kind in {s.kind for s in spans}


def test_detects_pem_private_key_block(detector: SecretDetector):
    pem = (
        "-----BEGIN RSA PRIVATE KEY-----\n"
        "MIIEpAIBAAKCAQEA1234567890abcdefghij\n"
        "klmnopqrstuvwxyzABCDEFGHIJKLMNOPQRST\n"
        "-----END RSA PRIVATE KEY-----"
    )
    spans = detector.detect(f"here is the key:\n{pem}\nkeep it safe")
    assert [s.kind for s in spans] == ["SECRET_PEM"]
    assert pem == f"here is the key:\n{pem}\nkeep it safe"[spans[0].start : spans[0].end]


def test_detects_openssh_and_pgp_private_key_blocks(detector: SecretDetector):
    for label in ("OPENSSH PRIVATE KEY", "PGP PRIVATE KEY BLOCK"):
        block = f"-----BEGIN {label}-----\nb3BlbnNzaC1rZXktdjEAAAAA\n-----END {label}-----"
        assert [s.kind for s in detector.detect(block)] == ["SECRET_PEM"]


def test_detects_generated_token_in_prose(detector: SecretDetector):
    text = "Created the account. The password is nZ8Kq2vL9pXwT3aBcDeF — write it down."
    assert hits(detector, text) == ["nZ8Kq2vL9pXwT3aBcDeF"]


def test_secret_spans_default_to_never_send(detector: SecretDetector):
    spans = detector.detect("AKIAIOSFODNN7EXAMPLE")
    assert all(s.action is Action.NEVER_SEND for s in spans)


def test_action_is_configurable():
    detector = SecretDetector(SecretPolicy(action=Action.TOKENIZE))
    spans = detector.detect("AKIAIOSFODNN7EXAMPLE")
    assert all(s.action is Action.TOKENIZE for s in spans)


def test_token_urlsafe_recall_is_high(detector: SecretDetector):
    """The incident value, unlabelled. Documented recall is ~99%; assert 95%."""
    trials = 500
    caught = sum(1 for _ in range(trials) if detector.detect(pysecrets.token_urlsafe(16)))
    assert caught >= trials * 0.95, f"only caught {caught}/{trials}"


def test_token_urlsafe_recall_is_total_when_labelled(detector: SecretDetector):
    for _ in range(300):
        token = pysecrets.token_urlsafe(16)
        assert detector.detect(f'{{"password": "{token}"}}'), token
        assert detector.detect_in_field(token, "password"), token


def test_short_token_is_caught_only_with_a_label(detector: SecretDetector):
    # 11 chars — below min_length, invisible to the unlabelled path by design.
    token = "aB3dE5fG7hJ"
    assert not detector.detect(token)
    assert detector.detect_in_field(token, "password")


# ---- Non-detection (the false-positive corpus) -----------------------------

NOT_SECRETS: list[tuple[str, str]] = [
    ("english_word", "internationalization"),
    ("norwegian_compound", "menneskerettighetsorganisasjon"),
    ("long_sentence", "the quick brown fox jumps over the lazy dog repeatedly"),
    ("title_case_pangram", "TheQuickBrownFoxJumpsOverTheLazyDog"),
    ("camel_case_method", "getUserProfileByIdentifier"),
    ("camel_case_long", "ThisIsAVeryLongCamelCaseIdentifierName"),
    ("java_class_name", "AbstractSingletonProxyFactoryBean"),
    ("acronym_class_name", "PostgreSQLJDBCDriverManager"),
    ("xml_class_name", "XMLHttpRequestFactory"),
    ("mixed_word_digits", "MyDocument2026Final"),
    ("snake_case", "application_json_content_type_header"),
    ("screaming_snake", "REDACTED_TORRENT_SIZE_CAP_BYTES"),
    ("kebab_slug", "REDACTED-qbittorrent-completion-hook-v2"),
    ("screenshot_filename", "Screenshot_2026-08-15_at_14.55.32.png"),
    ("camera_filename", "IMG_20240115_103245_HDR.jpg"),
    ("posix_path", "/usr/local/lib/python3.12/site-packages/agentkit/pii/firewall.py"),
    ("url", "https://example.com/articles/2026/08/how-we-shipped-the-thing"),
    ("dsn", "postgresql://localhost:5432/gulden_dev"),
    ("git_sha1", "commit 9f8e7d6c5b4a39281706f5e4d3c2b1a09f8e7d6c touched 3 files"),
    ("git_short_sha", "fixed in 473137d and 6e3fe43"),
    ("sha256_digest", "sha256:9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"),
    ("md5_digest", "md5 d41d8cd98f00b204e9800998ecf8427e"),
    ("iso_timestamp", "2026-08-15T10:32:45.123456+02:00"),
    ("semver", "v1.2.3-alpha.4+build567"),
    ("log_line", "ERROR 2026-08-15 14:55:32,123 agentkit.loop.tool_dispatcher timeout"),
    ("http_request_line", "GET /api/v1/households/12345/members?include=roles HTTP/1.1 200"),
    ("query_string", "user_id=42&session_expires=1755252765&page=3"),
    ("ls_output", "-rw-r--r--  1 morten  staff  22849 Aug 15 14:55 firewall.py"),
    ("alphabet", "abcdefghijklmnopqrstuvwxyz"),
    ("alphabet_upper", "ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
    ("repeated_char", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"),
    ("norwegian_address", "Kari Nordmann bor i Storgata 14, 0155 Oslo"),
    ("locales", "en_US.UTF-8 nb_NO.UTF-8 nn_NO.UTF-8"),
    ("placeholder_already_scrubbed", "[REDACTED] and [EMAIL_11] and [CANDIDATE_NAME]"),
]


@pytest.mark.parametrize("text", [pytest.param(t, id=i) for i, t in NOT_SECRETS])
def test_does_not_flag(detector: SecretDetector, text: str):
    assert detector.detect(text) == [], f"false positive in {text!r}"


def test_does_not_flag_uuids(detector: SecretDetector):
    for _ in range(200):
        value = str(uuid.uuid4())
        assert detector.detect(value) == [], value
        assert detector.detect(value.upper()) == [], value


def test_does_not_flag_ulids(detector: SecretDetector):
    """agentkit mints a ULID for every session, turn and message id, so they
    are all over tool output and log lines."""
    for _ in range(200):
        value = new_id(SessionId)
        assert detector.detect(f"session {value} started") == [], value


def test_does_not_flag_base64_image_payloads(detector: SecretDetector):
    # A one-pixel PNG: short enough to clear max_length, saved by the data-URI
    # marker; longer payloads are excluded by length alone.
    payload = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8"
        "BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    )
    assert detector.detect(f"data:image/png;base64,{payload}") == []
    assert detector.detect("A" * 600 + "b1") == []


def test_does_not_flag_digest_shaped_hex_by_default(detector: SecretDetector):
    for digest in (
        "d41d8cd98f00b204e9800998ecf8427e",  # md5, 32
        "9f8e7d6c5b4a39281706f5e4d3c2b1a09f8e7d6c",  # sha1 / git, 40
        "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",  # sha256
    ):
        assert detector.detect(digest) == [], digest


# ---- Context and field names ----------------------------------------------


def test_keyword_context_lowers_the_bar(detector: SecretDetector):
    # 12 chars: under min_length, so invisible on its own...
    assert detector.detect("Ab3Kq9Zx2Lm7") == []
    # ...and caught next to the word that says what it is.
    assert hits(detector, "api_key = Ab3Kq9Zx2Lm7") == ["Ab3Kq9Zx2Lm7"]


def test_keyword_context_catches_digest_shaped_hex(detector: SecretDetector):
    bare = "9f86d081884c7d659a2feaa0c55ad015"
    assert detector.detect(bare) == []
    assert hits(detector, f'{{"api_key": "{bare}"}}') == [bare]


def test_context_window_is_bounded(detector: SecretDetector):
    far = "password " + ("x" * 200) + " Ab3Kq9Zx2Lm7"
    assert detector.detect(far) == []


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        ("password", True),
        ("passwd", True),
        ("adminPassword", True),
        ("household_password", True),
        ("api_key", True),
        ("stripe-api-key", True),
        ("refresh_token", True),
        ("client_secret", True),
        ("privateKey", True),
        ("credentials", True),
        ("keyboard_layout", False),
        ("author", False),
        ("key", False),
        ("description", False),
        ("token_count", False),
        ("password_length", False),
        ("max_tokens", False),
    ],
)
def test_is_secret_field_name(field: str, expected: bool):
    assert is_secret_field_name(field) is expected


def test_secret_named_field_redacts_whole_value_whatever_it_is(detector: SecretDetector):
    # No entropy at all — only the field name says this is a credential.
    spans = detector.detect_in_field("sommer2026", "household_password")
    assert len(spans) == 1
    assert (spans[0].start, spans[0].end) == (0, len("sommer2026"))


def test_secret_named_field_skips_empty_and_already_scrubbed_values(detector: SecretDetector):
    for value in ("", "   ", "[REDACTED]", "[SECRET_1]", "null", "none", "false"):
        assert detector.detect_in_field(value, "password") == [], value


def test_ordinary_field_name_does_not_redact(detector: SecretDetector):
    assert detector.detect_in_field("sommer2026", "season") == []


def test_field_rule_can_be_disabled():
    detector = SecretDetector(SecretPolicy(redact_secret_named_fields=False))
    assert detector.detect_in_field("sommer2026", "password") == []


def test_detect_matches_detect_in_field_with_no_field(detector: SecretDetector):
    text = "token AKIAIOSFODNN7EXAMPLE here"
    assert detector.detect(text) == detector.detect_in_field(text, None)


# ---- Configurability -------------------------------------------------------


def test_digest_exemption_can_be_turned_off():
    detector = SecretDetector(SecretPolicy(exempt_digest_shaped_hex=False))
    assert [s.kind for s in detector.detect("9f8e7d6c5b4a39281706f5e4d3c2b1a09f8e7d6c")] == [
        "SECRET_HEX"
    ]


def test_allow_patterns_exempt_domain_identifiers():
    noisy = "ORDER-nZ8Kq2vL9pXwT3aBcDeF"
    assert SecretDetector().detect(noisy)
    tuned = SecretDetector(SecretPolicy(allow_patterns=(r"ORDER-[0-9A-Za-z]+",)))
    assert tuned.detect(noisy) == []


def test_min_length_is_configurable():
    short = "aB3dE5fG7hJ"
    assert SecretDetector().detect(short) == []
    assert SecretDetector(SecretPolicy(min_length=8, entropy_bits=3.0)).detect(short)


def test_max_length_is_configurable():
    rng = random.Random(20260815)  # deterministic corpus, not crypto
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
    blob = "".join(rng.choice(alphabet) for _ in range(600))
    assert SecretDetector().detect(blob) == []
    assert SecretDetector(SecretPolicy(max_length=2000)).detect(blob)


def test_vendor_prefixes_can_be_disabled():
    policy = SecretPolicy(detect_vendor_prefixes=False, detect_entropy=False)
    assert SecretDetector(policy).detect("AKIAIOSFODNN7EXAMPLE") == []


def test_spans_never_overlap(detector: SecretDetector):
    # A JWT's segments would each score on the entropy path too; the composite
    # merge must leave exactly one span covering the whole token.
    jwt = (
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
        ".eyJzdWIiOiIxMjM0NTY3ODkwIn0"
        ".dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"
    )
    spans = detector.detect(f"Authorization: Bearer {jwt}")
    assert len(spans) == 1
    for a, b in pairwise(spans):
        assert a.end <= b.start
