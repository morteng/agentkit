"""Configure structlog for the library.

Library code uses ``log = get_logger(__name__)`` and binds context (turn_id,
session_id) at call sites. Consumers can override the processor chain.
"""

import logging
from typing import Any

import structlog


def get_logger(name: str) -> structlog.BoundLogger:
    return structlog.get_logger(name)


def configure_default_logging(*, level: int = logging.INFO, json: bool = False) -> None:
    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
    ]
    if json:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(level),
        cache_logger_on_first_use=True,
    )
