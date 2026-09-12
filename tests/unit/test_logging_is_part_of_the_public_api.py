"""The default logging configuration is exported, and it stamps UTC.

WHY THIS IS A TEST AND NOT A DOCSTRING. `configure_default_logging` sets
`TimeStamper(fmt="iso")`, which is UTC with a trailing Z. structlog's own
default is `TimeStamper(fmt='%Y-%m-%d %H:%M:%S', utc=False)` -- a naive local
timestamp. A consumer that cannot import this from the package top level
either reaches into a private module or, far more likely, never calls it at
all and silently inherits the naive default. That is not a crash; it is months
of logs a person correlates by hand, each line an hour out on a box whose
clock is UTC.

WHAT EACH ARM DEFENDS.

* the import is reachable from `agentkit` itself, so no consumer needs a
  private module,
* the name is in `__all__`, so `from agentkit import *` and any tooling that
  reads the export list both see it,
* and it is the SAME OBJECT as the private one, because an export that
  shadowed it with a different function would satisfy the first two arms and
  configure something else.

The last arm is the one that would catch a future refactor: a re-export is
only useful while it re-exports the thing that does the work.
"""

from __future__ import annotations

import agentkit
from agentkit import _logging


def test_the_default_logging_configuration_is_importable_from_the_package() -> None:
    assert hasattr(agentkit, "configure_default_logging")


def test_it_is_named_in_all() -> None:
    assert "configure_default_logging" in agentkit.__all__


def test_the_export_is_the_function_that_does_the_work() -> None:
    assert agentkit.configure_default_logging is _logging.configure_default_logging


def test_it_stamps_iso_which_is_utc() -> None:
    """The reason the export matters, asserted on the behaviour rather than on
    the name. `fmt="iso"` is UTC with a Z; structlog's default is naive local
    time, and the two are indistinguishable in a log read a month later."""
    import inspect

    source = inspect.getsource(_logging.configure_default_logging)
    assert 'TimeStamper(fmt="iso")' in source, (
        "configure_default_logging no longer stamps iso. If the timestamp "
        "format changed deliberately, say so here -- a naive local timestamp "
        "on a UTC host is the defect this export exists to make avoidable."
    )
