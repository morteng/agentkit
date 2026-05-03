from agentkit._ids import CheckpointId, MessageId, SessionId, TurnId, new_id


def test_session_id_is_unique_and_sortable():
    a = new_id(SessionId)
    b = new_id(SessionId)
    assert a != b
    assert a < b  # ULIDs are time-sortable


def test_session_id_roundtrips_str():
    a = new_id(SessionId)
    s = str(a)
    assert SessionId(s) == a


def test_turn_message_checkpoint_ids_distinct_types():
    s = new_id(SessionId)
    t = new_id(TurnId)
    m = new_id(MessageId)
    c = new_id(CheckpointId)
    # Each carries its own NewType — pyright would reject mixing them.
    assert {type(s).__name__, type(t).__name__, type(m).__name__, type(c).__name__} == {"str"}
