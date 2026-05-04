import contextlib

from hypothesis import given, settings
from hypothesis import strategies as st

from agentkit.errors import InvalidPhaseTransition
from agentkit.loop.phase import TRANSITIONS, Phase, is_terminal, validate_transition


@given(start=st.sampled_from([p for p in Phase if not is_terminal(p)]))
@settings(max_examples=50)
def test_every_non_terminal_phase_has_at_least_one_outgoing(start):
    assert TRANSITIONS[start], f"{start} has no outgoing transitions"


@given(sequence=st.lists(st.sampled_from(list(Phase)), min_size=2, max_size=20))
@settings(max_examples=200)
def test_random_sequences_only_raise_invalid_phase_transition(sequence):
    for i in range(len(sequence) - 1):
        with contextlib.suppress(InvalidPhaseTransition):
            validate_transition(sequence[i], sequence[i + 1])


def _walk_paths(start: Phase, max_depth: int = 8) -> list[list[Phase]]:
    """All non-cycling paths from start to a terminal phase, capped at max_depth."""
    paths: list[list[Phase]] = []

    def dfs(node: Phase, path: list[Phase]) -> None:
        if is_terminal(node) or len(path) >= max_depth:
            paths.append(path[:])
            return
        for nxt in TRANSITIONS.get(node, frozenset()):
            if nxt in path:
                continue
            path.append(nxt)
            dfs(nxt, path)
            path.pop()

    dfs(start, [start])
    return paths


def test_every_path_from_idle_terminates():
    paths = _walk_paths(Phase.IDLE, max_depth=12)
    # At least one path must reach TURN_ENDED.
    assert any(Phase.TURN_ENDED in p for p in paths)
