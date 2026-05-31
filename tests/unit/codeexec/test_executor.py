import asyncio

import pytest

from agentkit.codeexec.executor import execute
from agentkit.codeexec.limits import ExecLimits
from agentkit.codeexec.namespace import SAFE_MODULES


async def test_safe_modules_usable_without_import_when_merged():
    res = await execute(
        {**SAFE_MODULES},
        "print(math.sqrt(16))\nreturn datetime.date(2026, 1, 1).isoformat()",
    )
    assert res.error is None
    assert res.stdout == "4.0\n"
    assert res.return_value == "2026-01-01"


async def test_injected_module_dunder_access_still_blocked():
    # Handing the real module object to the script must not reopen the escape:
    # the validator rejects dunder attribute access at parse time.
    res = await execute({**SAFE_MODULES}, "return math.__loader__")
    assert res.error_type == "CodeValidationError"
    assert res.return_value is None


async def test_runs_plain_script_and_captures_print():
    res = await execute({}, "print('hi')\nx = 2 + 3\nprint(x)")
    assert res.stdout == "hi\n5\n"
    assert res.error is None


async def test_returns_value():
    res = await execute({}, "return 41 + 1")
    assert res.return_value == 42


async def test_awaits_injected_async_function():
    async def fetch():
        return [1, 2, 3]

    res = await execute(
        {"fetch": fetch}, "rows = await fetch()\nprint(len(rows))\nreturn sum(rows)"
    )
    assert res.stdout == "3\n"
    assert res.return_value == 6


async def test_validation_error_surfaces():
    res = await execute({}, "import os")
    assert res.error is not None
    assert res.error_type == "CodeValidationError"


async def test_runtime_error_is_captured_not_raised():
    res = await execute({}, "x = 1 / 0")
    assert res.error_type == "ZeroDivisionError"
    assert res.return_value is None


async def test_timeout():
    async def slow():
        await asyncio.sleep(5)

    res = await execute({"slow": slow}, "await slow()", ExecLimits(wall_clock_s=0.1))
    assert res.error_type == "CodeTimeoutError"


async def test_injected_systemexit_is_captured_not_propagated():
    async def boom():
        raise SystemExit(42)

    res = await execute({"boom": boom}, "await boom()")
    assert res.error_type == "SystemExit"
    assert res.return_value is None


async def test_injected_keyboardinterrupt_is_captured():
    async def boom():
        raise KeyboardInterrupt()

    res = await execute({"boom": boom}, "await boom()")
    assert res.error_type == "KeyboardInterrupt"


async def test_cancellederror_propagates_and_is_not_captured():
    async def boom():
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await execute({"boom": boom}, "await boom()")
