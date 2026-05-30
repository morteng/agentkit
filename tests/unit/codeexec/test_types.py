from agentkit.codeexec.errors import (
    CodeExecutionError,
    CodeTimeoutError,
    CodeValidationError,
)
from agentkit.codeexec.limits import ExecLimits


def test_exec_limits_defaults():
    lim = ExecLimits()
    assert lim.wall_clock_s == 30.0
    assert lim.max_stdout_bytes == 64 * 1024


def test_errors_are_distinct_subclasses():
    assert issubclass(CodeValidationError, CodeExecutionError)
    assert issubclass(CodeTimeoutError, CodeExecutionError)
    assert not issubclass(CodeValidationError, CodeTimeoutError)
