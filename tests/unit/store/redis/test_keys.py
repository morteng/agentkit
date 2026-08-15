import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from agentkit._ids import CheckpointId, OwnerId, SessionId
from agentkit.store.memory import MemoryScope
from agentkit.store.redis.keys import (
    MAX_MEMORY_KEY_LENGTH,
    KeyBuilder,
    escape_key_part,
    validate_memory_key,
)


def test_session_key_includes_prefix():
    kb = KeyBuilder(prefix="agentkit")
    sid = SessionId("01H...A")
    assert kb.session(sid) == "agentkit:sess:01H...A"


def test_messages_key_distinct_from_session_key():
    kb = KeyBuilder(prefix="agentkit")
    sid = SessionId("01H...A")
    assert kb.session(sid) != kb.messages(sid)


def test_owner_index_key_escapes_the_owner_id():
    """``u:1`` is the house style for owner ids, so the escaping shows up here
    first: the colon belongs to the *value*, not to the key structure."""
    kb = KeyBuilder(prefix="agentkit")
    assert kb.owner_index(OwnerId("u:1")) == "agentkit:owner:u%3A1:sessions"


def test_memory_key_includes_full_scope():
    kb = KeyBuilder(prefix="agentkit")
    scope = MemoryScope(namespace="globex", user_id="u1")
    assert "globex" in kb.memory(scope, "k1")
    assert "u1" in kb.memory(scope, "k1")


def test_checkpoint_key():
    kb = KeyBuilder(prefix="agentkit")
    cid = CheckpointId("01H...C")
    assert kb.checkpoint(cid) == "agentkit:ckpt:01H...C"


# ---- escaping: the model-controlled segments --------------------------------


def test_memory_key_with_colon_cannot_reach_another_scope():
    """The headline attack: a memory key that pretends to be scope structure.

    Saving under key ``"u:victim:secret"`` in the bare namespace must not write
    into the ``user_id="victim"`` scope's namespace.
    """
    kb = KeyBuilder(prefix="agentkit")
    attacker = MemoryScope(namespace="ns")
    victim = MemoryScope(namespace="ns", user_id="victim")
    assert kb.memory(attacker, "u:victim:secret") != kb.memory(victim, "secret")


def test_memory_key_cannot_clobber_the_scope_index():
    """``_index`` used to be an ordinary, writable key name."""
    kb = KeyBuilder(prefix="agentkit")
    scope = MemoryScope(namespace="ns")
    for hostile in ("_index", "%index", "%25index"):
        assert kb.memory(scope, hostile) != kb.memory_index(scope)


def test_scope_component_with_colon_cannot_forge_another_scope():
    """Escaping must apply to the scope too — a tenant id is often as
    externally-derived as the key is."""
    kb = KeyBuilder(prefix="agentkit")
    forged = MemoryScope(namespace="ns:u:victim")
    real = MemoryScope(namespace="ns", user_id="victim")
    assert kb.memory(forged, "k") != kb.memory(real, "k")


def test_percent_itself_is_escaped_so_the_encoding_stays_injective():
    assert escape_key_part("%3A") != escape_key_part(":")


# ---- property-style: collisions are impossible, not just unlikely -----------

_TEXT = st.text(min_size=1, max_size=12)
_OPT_TEXT = st.none() | _TEXT
_SCOPES = st.builds(
    MemoryScope,
    namespace=_TEXT,
    tenant_id=_OPT_TEXT,
    user_id=_OPT_TEXT,
    session_id=_OPT_TEXT,
)


@given(scope=_SCOPES, key=_TEXT)
@settings(max_examples=300)
def test_no_user_key_can_ever_equal_an_index_key(scope, key):
    """For every scope and every key a model can produce, the memory key and
    that scope's index key are distinct."""
    kb = KeyBuilder(prefix="agentkit")
    assert kb.memory(scope, key) != kb.memory_index(scope)


@given(scope_a=_SCOPES, scope_b=_SCOPES, key_a=_TEXT, key_b=_TEXT)
@settings(max_examples=500)
def test_distinct_scope_key_pairs_never_share_a_redis_key(scope_a, scope_b, key_a, key_b):
    """Cross-scope isolation as a property: the (scope, key) -> redis key map is
    injective, so no key written in one scope is readable from another."""
    kb = KeyBuilder(prefix="agentkit")
    same_input = (scope_a, key_a) == (scope_b, key_b)
    same_output = kb.memory(scope_a, key_a) == kb.memory(scope_b, key_b)
    assert same_output == same_input


@given(scope_a=_SCOPES, scope_b=_SCOPES, key=_TEXT)
@settings(max_examples=300)
def test_a_key_in_one_scope_never_lands_on_another_scopes_index(scope_a, scope_b, key):
    kb = KeyBuilder(prefix="agentkit")
    assert kb.memory(scope_a, key) != kb.memory_index(scope_b)


@given(scope_a=_SCOPES, scope_b=_SCOPES)
@settings(max_examples=300)
def test_distinct_scopes_have_distinct_index_keys(scope_a, scope_b):
    kb = KeyBuilder(prefix="agentkit")
    assert (kb.memory_index(scope_a) == kb.memory_index(scope_b)) == (scope_a == scope_b)


@given(owner_a=_TEXT, owner_b=_TEXT)
@settings(max_examples=200)
def test_distinct_owners_have_distinct_owner_indexes(owner_a, owner_b):
    kb = KeyBuilder(prefix="agentkit")
    built_a = kb.owner_index(OwnerId(owner_a))
    built_b = kb.owner_index(OwnerId(owner_b))
    assert (built_a == built_b) == (owner_a == owner_b)


@given(sid=_TEXT)
@settings(max_examples=200)
def test_per_session_keys_never_collide_with_each_other(sid):
    kb = KeyBuilder(prefix="agentkit")
    session_id = SessionId(sid)
    built = {
        kb.session(session_id),
        kb.messages(session_id),
        kb.event_channel(session_id),
        kb.event_buffer(session_id),
    }
    assert len(built) == 4


# ---- write-path validation --------------------------------------------------


@pytest.mark.parametrize("bad", ["", "   ", "\t\n"])
def test_validate_memory_key_rejects_empty_keys(bad):
    with pytest.raises(ValueError, match="non-empty"):
        validate_memory_key(bad)


def test_validate_memory_key_rejects_oversized_keys():
    with pytest.raises(ValueError, match="too long"):
        validate_memory_key("k" * (MAX_MEMORY_KEY_LENGTH + 1))


def test_validate_memory_key_accepts_hostile_but_reasonable_keys():
    """Escaping — not rejection — is what makes ``:`` safe. The validator must
    not start refusing legitimate keys that merely look structural."""
    validate_memory_key("u:1:preferences")
    validate_memory_key("_index")
