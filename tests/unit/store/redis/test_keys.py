from agentkit._ids import CheckpointId, OwnerId, SessionId
from agentkit.store.memory import MemoryScope
from agentkit.store.redis.keys import KeyBuilder


def test_session_key_includes_prefix():
    kb = KeyBuilder(prefix="agentkit")
    sid = SessionId("01H...A")
    assert kb.session(sid) == "agentkit:sess:01H...A"


def test_messages_key_distinct_from_session_key():
    kb = KeyBuilder(prefix="agentkit")
    sid = SessionId("01H...A")
    assert kb.session(sid) != kb.messages(sid)


def test_owner_index_key():
    kb = KeyBuilder(prefix="agentkit")
    assert kb.owner_index(OwnerId("u:1")) == "agentkit:owner:u:1:sessions"


def test_memory_key_includes_full_scope():
    kb = KeyBuilder(prefix="agentkit")
    scope = MemoryScope(namespace="ampaera", user_id="u1")
    assert "ampaera" in kb.memory(scope, "k1")
    assert "u1" in kb.memory(scope, "k1")


def test_checkpoint_key():
    kb = KeyBuilder(prefix="agentkit")
    cid = CheckpointId("01H...C")
    assert kb.checkpoint(cid) == "agentkit:ckpt:01H...C"
