from agentkit.store.redis.serialization import from_versioned_json, to_versioned_json


def test_versioned_round_trip():
    payload = {"hello": "world"}
    raw = to_versioned_json(payload, schema_version=1)
    parsed, version = from_versioned_json(raw)
    assert parsed == payload
    assert version == 1


def test_old_version_returns_old_version_number():
    raw = b'{"_v": 0, "hello": "world"}'
    parsed, version = from_versioned_json(raw)
    assert parsed == {"hello": "world"}
    assert version == 0


def test_unversioned_payload_is_treated_as_v0():
    raw = b'{"hello": "world"}'
    parsed, version = from_versioned_json(raw)
    assert parsed == {"hello": "world"}
    assert version == 0
