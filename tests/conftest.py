"""Shared pytest configuration."""

import asyncio
import logging

import pytest


@pytest.fixture(autouse=True)
def _silence_third_party_logs(caplog):
    """Quiet third-party loggers in tests; keep agentkit logs visible."""
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    yield


@pytest.fixture
def event_loop_policy():
    """Pin to default asyncio policy for deterministic teardown."""
    return asyncio.DefaultEventLoopPolicy()
