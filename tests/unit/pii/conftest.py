"""Fake Detector / TokenMap for firewall unit tests.

agentkit ships no detector/tokenmap — consumers supply them. These fakes model
the contract: a keyword detector (fnr → NEVER_SEND, name/email/phone → TOKENIZE)
and a deterministic per-value token map.
"""

import re
from datetime import UTC, datetime

import pytest

from agentkit._content import TextBlock
from agentkit._ids import MessageId, SessionId, new_id
from agentkit._messages import Message, MessageRole
from agentkit.pii.types import Action, Span


class FakeDetector:
    """Regex keyword detector. NEVER_SEND for fnr-like 11-digit runs; TOKENIZE
    for a fixed set of known literals."""

    def __init__(self) -> None:
        # kind -> (regex, action)
        self._patterns: list[tuple[str, re.Pattern[str], Action]] = [
            ("FNR", re.compile(r"\b\d{11}\b"), Action.NEVER_SEND),
            ("EMAIL", re.compile(r"\b[\w.]+@[\w.]+\.\w+\b"), Action.TOKENIZE),
            ("NAME", re.compile(r"\bKari Nordmann\b"), Action.TOKENIZE),
            ("NAME", re.compile(r"\bOla Nordmann\b"), Action.TOKENIZE),
            ("PHONE", re.compile(r"\b\d{8}\b"), Action.TOKENIZE),
        ]

    def detect(self, text: str) -> list[Span]:
        spans: list[Span] = []
        for kind, pat, action in self._patterns:
            for m in pat.finditer(text):
                spans.append(Span(start=m.start(), end=m.end(), kind=kind, action=action))
        return spans


class FakeTokenMap:
    """Deterministic per-value token map. Same value → same token forever."""

    def __init__(self) -> None:
        self._to_token: dict[str, str] = {}
        self._to_value: dict[str, str] = {}
        self._counters: dict[str, int] = {}

    def token_for(self, value: str, kind: str) -> str:
        if value in self._to_token:
            return self._to_token[value]
        n = self._counters.get(kind, 0) + 1
        self._counters[kind] = n
        token = f"[{kind}_{n}]"
        self._to_token[value] = token
        self._to_value[token] = value
        return token

    def value_for(self, token: str) -> str | None:
        return self._to_value.get(token)

    def all_tokens(self) -> set[str]:
        return set(self._to_value)


@pytest.fixture
def detector() -> FakeDetector:
    return FakeDetector()


@pytest.fixture
def tmap() -> FakeTokenMap:
    return FakeTokenMap()


def make_msg(role: MessageRole, *blocks: object) -> Message:
    return Message(
        id=new_id(MessageId),
        session_id=new_id(SessionId),
        role=role,
        content=list(blocks),  # type: ignore[arg-type]
        created_at=datetime.now(UTC),
    )


def text_block(text: str) -> TextBlock:
    return TextBlock(text=text)
